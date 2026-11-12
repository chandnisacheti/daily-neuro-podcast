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
SPEED = 0.92  # playback speed of the final mix; 15% slower felt too slow, she asked for 8% faster (2026-11-11)
HOSTS = ("Nora", "Theo")
# Distinct voices: Nora bright and quick, Theo lower and relaxed. Fallback pair if Google renames voices.
VOICES = {"Nora": "Aoede", "Theo": "Charon"}
FALLBACK_VOICES = {"Nora": "Kore", "Theo": "Puck"}
STYLES = {
    "Nora": "a real person chatting on a casual podcast with a friend, not a narrator or announcer; warm and bright, genuinely curious, smiles while talking; uneven natural pacing, speeds up when excited and slows on key points, small breaths, stresses the one important word in a sentence, reacts with real surprise or amusement",
    "Theo": "a real person chatting on a casual podcast with a friend, not a narrator; relaxed, lower-pitched, thoughtful engineer; thinks out loud, slight hesitations before a key point, dry humor with a hint of a laugh, varies pitch and pace like real conversation, never reads in a flat even rhythm",
}
EDGE_VOICES = {"Nora": "en-US-AvaNeural", "Theo": "en-US-AndrewNeural"}
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
        # Backups when the newest model is overloaded (503s killed the 2026-11-10 run).
        TEXT_FALLBACKS[:] = [n for n in text[1:6]] + ["gemini-flash-latest"]
    if not TTS_MODEL:
        tts = [n for n in names if "tts" in n]
        tts.sort(key=lambda n: (_ver(n), "flash" in n, "lite" not in n), reverse=True)
        TTS_MODEL = tts[0] if tts else ""
    log(f"using text model {TEXT_MODEL}, tts model {TTS_MODEL or '(none, Edge voices)'}")
DRY = os.environ.get("DRY_RUN") == "1"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------- 1. news ----------
SOURCE_STATUS = []


def gather_news(hours=30):
    cutoff = time.time() - hours * 3600
    seen, items = set(), []
    SOURCE_STATUS.clear()
    for label, url, cap in FEEDS:
        try:
            r = requests.get(url, timeout=20, headers={"User-Agent": UA, "Accept": "application/rss+xml, application/xml, text/xml, */*"})
            r.raise_for_status()
            parsed = feedparser.parse(r.content)
        except Exception as e:  # noqa: BLE001
            log(f"skip {label}: {e}")
            SOURCE_STATUS.append((label, f"failed ({str(e)[:60]})"))
            continue
        n = 0
        for e in parsed.entries:
            # Use the original publish date, not "updated" (feeds bump old stories when they edit them).
            # No date at all = can't prove it's fresh, so skip it (Chandni got 6-day-old news, 2026-11-10).
            t = e.get("published_parsed") or e.get("updated_parsed")
            if not t or calendar.timegm(t) < cutoff:
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
        SOURCE_STATUS.append((label, f"{n} new items"))
    for i, it in enumerate(items):
        it["id"] = i + 1
    return items


# ---------- 2. script ----------
TEXT_FALLBACKS = []


def gemini(model, body, tries=4):
    """Call Gemini with retries; if the text model stays overloaded, fall back to other Flash models."""
    try:
        return _gemini(model, body, tries)
    except RuntimeError:
        if model != TEXT_MODEL:
            raise
        for alt in TEXT_FALLBACKS:
            if alt == model:
                continue
            log(f"falling back to {alt}")
            try:
                return _gemini(alt, body, 2)
            except RuntimeError:
                continue
        raise


def _gemini(model, body, tries=4):
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
        if r.status_code == 429 and ("PerDay" in r.text or "per_day" in r.text.lower() or i >= 1):
            # Daily free quota used up: retrying only burns more of it, so move on to another model.
            break
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(30 * (i + 1))
            continue
        break
    raise RuntimeError(f"Gemini call to {model} failed")


EDITOR_PROMPT = """You are the editor of "Neuro Morning", a daily ~10 minute audio news briefing for Chandni, an
MS Biomedical Engineering student at Columbia (NYC) working on biosignal ML (EEG, EMG, HRV, EDA) and
closed-loop neuromodulation, who wants a career in neurotech, wearables, digital health and health tech.

Pick the stories for today's episode from the numbered news items, and plan the rundown.

Selection rules:
- FIVE main stories by default: the THREE biggest neurotech stories of the day (BCI, neuromodulation,
  neural recording, EEG/EMG and neural wearables, neuro research) and the TWO biggest stories from
  outside neurotech (digital health, health tech, medtech, wearables). Wearables can count on either
  side. Go to 4 or 6 only if the day is unusually thin or something huge happened; never more than 6.
  Pick substantive stories with real depth, because each one gets a full segment.
- Prefer: funding, acquisitions, FDA decisions, notable research, product launches, NYC companies,
  and anything that signals hiring or growth.
- When several outlets cover the same story, that is a SIGNAL it matters: merge them into one story
  (use the best item's id; list the others in skipped as "merged into story N"). Never drop a story
  just because it has duplicates.
- Chandni especially wants: controversies and backlash over the scientific claims of neuro or health
  wearables and devices (validation disputes, critics vs founders), IPOs, big funding, FDA decisions,
  new wearables, and moves by well-known founders or companies. Treat these as strong picks even when
  the company is consumer-facing or outside the US.
- Skip:  items not about health, medicine or neuroscience (stock tickers,
  unrelated companies), thin press releases, shopping guides and generic gadget reviews, and stories already
  covered in recent episodes (listed below) unless there is a genuinely new development.
- Each story appears ONCE in the whole episode. Never assign the same story to two sections.
- Choose stories that connect to each other where possible, so the Trends segment has something real
  to tie together.

Sections are flexible. Default order: "Who Expanded" (funding, deals, FDA, growth), "New Research and
Products", "NYC Watch", "Trends", "Career Signal", "Recap". Drop a section if it would only repeat
earlier stories (for example, if the only NYC story was already covered, mention the NYC angle when
that story is told and skip NYC Watch). Rename or reorder sections when that makes the day clearer.
Trends, Career Signal and Recap tie together stories already told and must not introduce repeats as
new stories.

FRESHNESS: this is a DAILY show. Only pick news that first became public in the last 2 days. Skip an item
if it is a follow-up, analysis or re-report of an announcement made earlier (look for words like "earlier
this month", "last week", or a known older event), and log it in skipped as "old news".

Recently covered stories (do not repeat): {recent}

Return JSON:
{{"title": "short episode title",
  "sections": [{{"name": "...", "stories": [{{"id": 3, "angle": "one line: what to emphasize and why it matters to Chandni"}}]}}],
  "trends": ["1 or 2 BIG-PICTURE shifts in the industry right now, each one sentence of insight that is NOT a restatement of a story; cite today's stories only as examples (by id)"],
  "career_signal": "1 or 2 sentences: one non-obvious takeaway for Chandni's career (a skill, a kind of company, or a move to make this week), not a list of the companies just mentioned",
  "recap": ["exactly one memorable line per main story, in the order they were told"],
  "skipped": [{{"id": 7, "reason": "short reason"}}]}}
Every item id must appear exactly once, either in a section or in skipped.

NEWS ITEMS (JSON):
{items}
"""

WRITER_PROMPT = """Write the script for today's "Neuro Morning" from the editor's plan below.
Today is {date}. Listener: Chandni (see plan). She listens while walking or commuting.

HOSTS (distinct people, distinct ways of talking):
- Nora: the anchor. Quick, bright, curious. Short punchy sentences, asks the questions a smart
  listener would ask ("Wait, how big is that?"), moves the show along, announces sections.
- Theo: the engineer-analyst. Slower, thoughtful, a little dry humor. Explains the mechanism or the
  "so what" in plain words, often starts with "Here's the thing" or "So, practically..." (vary it).
They trade off who introduces stories. They sound like two sharp friends who both work in neurotech
and health tech, talking over coffee: smart and in the field, but relaxed. The listener should feel like
she is getting a high-value coffee chat with two insiders: real opinions, why a move is smart or risky,
what people in the industry are saying, what they would watch next. Talk the way smart people actually
talk, not in polished written sentences: contractions, short sentences, fragments where natural, quick
reactions ("Huh.", "Right.", "Wait, really?", "Okay, that's big."), friendly back-and-forth, and
the occasional half-finished thought that the other host picks up. Field vocabulary is fine when it is
the real word for the thing; just say it naturally and explain anything a smart outsider would not know.
Words like "landscape" or "underscores" are okay once in a while, but never stack them, and never sound
like a press release, a LinkedIn post or an essay. Never use "it's worth noting", "in an era of", "delve",
"paradigm". Never the same sentence pattern twice in a row.

HUMAN PACING (the voice engine reads punctuation as intonation):
- Use commas, ellipses ("so... here's the thing") and dashes for natural pauses and thinking beats.
- Let hosts start with small spoken openers sometimes ("So.", "Okay,", "Honestly,", "Hmm,").
- Ask real questions with question marks, and react with short lines, so the rhythm goes up and down.
- Sound like real speech, not a script: a light sprinkle of natural fillers and self-corrections
  (about one every four or five lines: "I mean,", "you know,", "kind of", "wait, actually..."),
  the occasional overlap where one host finishes the other's thought, and a laugh line now and then
  written as words ("Ha, okay, fair."). Never overdo it; clarity comes first.
- One thought per line. Never more than two sentences in a row from the same host without a reaction. Do not say "This matters because" or "The takeaway"
every time; vary how you land each story, but every story should end with a clear one-line point.

EASY TO FOLLOW BY EAR:
- Intro: two short lines (hello, the date), plus one line saying how many stories are coming
  ("Five stories today: three in neurotech, two in health tech"). No list of headlines up front.
- Number the stories as you go ("Story two of five"), so the listener always knows where she is.
- Announce each section clearly by name, e.g. "Okay, first up: who expanded this week." Signal
  transitions ("That's the deals. Now, research.").
- One story at a time: who, what happened, why it matters. Spell numbers the way people say them.
- PLAIN ENGLISH, built for the ear. Open every story with one short sentence a smart friend outside
  the industry would understand ("Philips is paying the government to settle a probe into its recalled
  sleep machines."). Then give one line of background if the story depends on history. Keep sentences
  under about 20 words, one idea per sentence. Replace jargon with plain words, or explain it in the
  same breath ("a consent decree, basically a court order"). Never stack legal, financial or chemical
  terms in one sentence, and use at most one number per sentence. If a story is mostly legal or
  financial, explain what it means for patients, products or the field, not the legal mechanics.
- Follow the plan's sections and order exactly; each story is told once.
- Give each main story a full segment (about 150 to 190 words): what happened, how it works, why it
  matters, and a clear one-line point at the end. Spend a little more of each segment on the hosts'
  own critical thinking about the news: what it signals about where the industry is heading, who it
  puts pressure on, whether the bet is smart or risky, or the open question nobody has answered yet.
  This is opinion and analysis, clearly framed as their take ("My read is...", "The question I'd ask
  is..."), never new facts.
- The closing segments are SHORT and must add something new. Never repeat the stories' facts.
  * Trends (about 60 to 80 words): one or two big-picture shifts happening in the industry right now.
    Say the insight first, then use one of today's stories as a quick example. It should make her
    think, not re-hear the episode.
  * Career Signal (about 40 to 60 words): one non-obvious takeaway for someone with Chandni's
    background (biosignal ML, EEG/EMG/HRV, closed-loop neuromodulation, productization): a skill to
    build, a kind of company to watch, or a move to make this week. No list of companies.
  * Recap (about 60 to 80 words): one short line per story in order ("One: ... Two: ..."), then
    one line that ties the day together.
- Use ONLY facts in each item's "facts" (a fact sheet verified with Google Search against credible
  sources) or its title and summary. Do not invent numbers, names, quotes, product details or context.
  If a detail is listed under UNCONFIRMED or is missing, leave it out or say plainly that it isn't
  confirmed yet. Never fill gaps with what sounds likely.
- Never guess what a product looks like or how it works. Describe form factor (ring, patch, headband,
  implant, app), price, sensors or specs ONLY if the item text says so; otherwise just say "wearable"
  or "device". A confident wrong detail is worse than a vague right one.
- About 1,300 to 1,450 words (this lands near 11 to 12 minutes).

Output ONLY dialogue lines, each starting with "Nora:" or "Theo:". No stage directions, no markdown.

EDITOR'S PLAN (JSON):
{plan}

ITEMS USED (JSON):
{items}
"""


def load_recent_titles(days=3):
    path = DOCS / "covered.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [t for d in data[-days:] for t in d.get("titles", [])]


def save_covered(slug, titles):
    path = DOCS / "covered.json"
    data = json.loads(path.read_text()) if path.exists() else []
    data = [d for d in data if d.get("slug") != slug] + [{"slug": slug, "titles": titles}]
    path.write_text(json.dumps(data[-7:], indent=1, ensure_ascii=False))


def plan_episode(items):
    if DRY:
        return {"title": "Test episode", "sections": [{"name": "Who Expanded", "stories": [{"id": it["id"], "angle": "test"} for it in items[:1]]}],
                "trends": [], "career_signal": "", "skipped": [{"id": it["id"], "reason": "test"} for it in items[1:]]}
    recent = load_recent_titles() or ["(none)"]
    slim = [{k: it[k] for k in ("id", "source", "title", "summary")} for it in items[:120]]
    body = {"contents": [{"parts": [{"text": EDITOR_PROMPT.format(recent=json.dumps(recent, ensure_ascii=False),
                                                                    items=json.dumps(slim, ensure_ascii=False))}]}],
            "generationConfig": {"temperature": 0.3, "responseMimeType": "application/json"}}
    out = gemini(TEXT_MODEL, body)
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    plan = json.loads(text)
    # enforce: each story once
    used = set()
    for sec in plan.get("sections", []):
        sec["stories"] = [st for st in sec.get("stories", []) if st.get("id") not in used and not used.add(st.get("id"))]
    plan["sections"] = [sec for sec in plan.get("sections", []) if sec["stories"] or
                        sec.get("name", "").lower() in ("trends", "career signal", "recap")]
    return plan


def write_script(items, plan, date):
    if DRY:
        return plan.get("title", "Test episode"), [("Nora", "Good morning! This is a test."), ("Theo", "Everything works.")]
    by_id = {it["id"]: it for it in items}
    used_ids = [st["id"] for sec in plan["sections"] for st in sec["stories"]]
    used_items = [by_id[i] for i in used_ids if i in by_id]
    body = {"contents": [{"parts": [{"text": WRITER_PROMPT.format(date=date, plan=json.dumps(plan, ensure_ascii=False),
                                                                    items=json.dumps(used_items, ensure_ascii=False))}]}],
            "generationConfig": {"temperature": 0.9}}
    out = gemini(TEXT_MODEL, body)
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    title, lines = plan.get("title") or f"Neuro Morning, {date}", []
    for raw in text.splitlines():
        raw = raw.strip().replace("**", "")
        m = re.match(r"^(Nora|Theo)\s*:\s*(.+)$", raw)
        if m:
            lines.append((m.group(1), m.group(2)))
    if len(lines) < 6:
        raise RuntimeError("Script came back too short:\n" + text[:1000])
    return title, lines



# ---------- 2b. research + fact check (no guessing) ----------
RESEARCH_PROMPT = """You are a fact-checker for a daily audio news briefing. Use Google Search to verify and
complete the facts for ONE news story. Prefer credible sources: the original reporting outlet, the company's
own press release or website, FDA or other regulators, peer-reviewed papers, and major outlets (STAT, Reuters,
Bloomberg, TechCrunch, Fierce, MedTech Dive). Ignore SEO blogs, content farms and AI-written sites.

Write a fact sheet of 6 to 12 short, plain bullet facts. Each bullet ends with the source outlet in
parentheses. Cover: what happened and when; who (company, people, roles); what the product or study
actually is, including its physical form if it is a device (ring, patch, headband, implant, app);
key numbers (money, accuracy, sample size, price) exactly as reported; one or two lines of background.
If sources disagree, say so. Then write a line starting "FIRST REPORTED:" with the date (YYYY-MM-DD) the
news itself first became public, from the earliest credible report or press release (not the date of a
later follow-up article). Then write a final line starting "UNCONFIRMED:" listing any important detail
you could not confirm from a credible source (or "none").

STORY: {title}
SOURCE GIVEN: {source}
SUMMARY GIVEN: {summary}
LINK: {link}
"""


def research_story(it):
    """Ground one story with Google Search. Returns (facts_text, [(title, url), ...])."""
    body = {"contents": [{"parts": [{"text": RESEARCH_PROMPT.format(**{k: it.get(k, "") for k in ("title", "source", "summary", "link")})}]}],
            "tools": [{"google_search": {}}],
            "generationConfig": {"temperature": 0.2}}
    # One try, no model fallback: Google Search grounding has its own small free quota, and retrying a
    # 429 here burned the whole 30-minute job on 2026-11-10.
    out = _gemini(TEXT_MODEL, body, tries=1)
    cand = out["candidates"][0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    chunks_ = (cand.get("groundingMetadata") or {}).get("groundingChunks") or []
    sources, seen = [], set()
    for c in chunks_:
        w = c.get("web") or {}
        if w.get("uri") and w["uri"] not in seen:
            seen.add(w["uri"]); sources.append((w.get("title", "") or w["uri"], w["uri"]))
    return text.strip(), sources


RESEARCH_STATE = {"off": False, "fails": 0}
MAX_NEWS_AGE_DAYS = 3  # a story first reported earlier than this is old news and gets dropped


def first_reported(facts):
    m = re.search(r"FIRST REPORTED:\s*\**\s*(\d{4}-\d{2}-\d{2})", facts or "")
    if not m:
        return None
    try:
        return dt.date.fromisoformat(m.group(1))
    except ValueError:
        return None


def research_stories(items, plan):
    """Attach a search-verified fact sheet to every story the editor picked, and drop old news."""
    if DRY:
        return
    by_id = {it["id"]: it for it in items}
    today = dt.datetime.now(dt.timezone.utc).date()
    for sec in plan.get("sections", []):
        keep = []
        for st in sec.get("stories", []):
            it = by_id.get(st.get("id"))
            if not it:
                keep.append(st)
                continue
            if not RESEARCH_STATE["off"]:
                try:
                    facts, sources = research_story(it)
                    if facts:
                        it["facts"], it["fact_sources"] = facts, sources
                        log(f"researched: {it['title'][:70]} ({len(sources)} sources)")
                    RESEARCH_STATE["fails"] = 0
                except Exception as e:  # noqa: BLE001
                    log(f"research failed for {it['title'][:70]}: {e}")
                    RESEARCH_STATE["fails"] += 1
                    if RESEARCH_STATE["fails"] >= 2:
                        RESEARCH_STATE["off"] = True
                        log("research: search quota looks used up; skipping the rest (writer sticks to the feed text)")
                time.sleep(6)
            first = first_reported(it.get("facts"))
            if first and (today - first).days > MAX_NEWS_AGE_DAYS:
                log(f"dropped old news ({first}): {it['title'][:70]}")
                plan.setdefault("do_not_mention", []).append(it["title"])
                plan.setdefault("skipped", []).append(
                    {"id": it["id"], "reason": f"old news: first reported {first.isoformat()}, more than {MAX_NEWS_AGE_DAYS} days ago"})
                continue
            keep.append(st)
        sec["stories"] = keep
    plan["sections"] = [sec for sec in plan.get("sections", []) if sec.get("stories") or
                        sec.get("name", "").lower() in ("trends", "career signal", "recap")]


FACTCHECK_PROMPT = """You are the fact-checker for a two-host news podcast script. Compare EVERY factual claim
in the script with the verified fact sheets below (and the original item text). A claim is supported only if
the fact sheet or item states it. Fix or cut anything else: invented details, guessed product forms (for
example calling a patch a "ring"), wrong or made-up numbers, names, dates, quotes, or overstated certainty.
When something important is unconfirmed, the hosts may say it is not yet confirmed; they never fill the gap.
Opinions, analysis and "what this could mean" lines are fine if they are framed as interpretation.

Keep the script's structure, voice, numbering, sections and length. Change as little as possible.
Output the full corrected script as dialogue lines starting with "Nora:" or "Theo:". Then a line
"CHANGES:" followed by one bullet per fix (what was wrong, what it became), or "- none".

VERIFIED STORIES (JSON):
{stories}

SCRIPT:
{script}
"""


def fact_check(lines, items, plan):
    """Second pass that removes anything the sources don't support. Returns (lines, changes)."""
    if DRY:
        return lines, []
    by_id = {it["id"]: it for it in items}
    used = [by_id[st["id"]] for sec in plan.get("sections", []) for st in sec.get("stories", []) if st.get("id") in by_id]
    stories = [{k: it.get(k) for k in ("title", "source", "summary", "facts")} for it in used]
    script = "\n".join(f"{s}: {t}" for s, t in lines)
    body = {"contents": [{"parts": [{"text": FACTCHECK_PROMPT.format(stories=json.dumps(stories, ensure_ascii=False), script=script)}]}],
            "generationConfig": {"temperature": 0.1}}
    try:
        out = gemini(TEXT_MODEL, body, tries=3)
    except Exception as e:  # noqa: BLE001
        log(f"fact check failed ({e}); keeping the writer's script")
        return lines, [f"Fact check could not run today ({e})."]
    text = "".join(p.get("text", "") for p in out["candidates"][0]["content"]["parts"])
    head, _, tail = text.partition("CHANGES:")
    fixed = []
    for raw in head.splitlines():
        m = re.match(r"^(Nora|Theo)\s*:\s*(.+)$", raw.strip().replace("**", ""))
        if m:
            fixed.append((m.group(1), m.group(2)))
    changes = [c.strip(" -*") for c in tail.splitlines() if c.strip(" -*") and c.strip(" -*").lower() != "none"]
    if len(fixed) < 0.7 * len(lines):
        log("fact check returned a much shorter script; keeping the writer's script")
        return lines, ["Fact check output looked incomplete, so the writer's script was kept."]
    return fixed, changes


def write_editor_log(items, plan, slug, date_str, fixes=()):
    by_id = {it["id"]: it for it in items}
    out = [f"# Editor's log, {date_str}", "", f"Episode: {plan.get('title', '')}", "",
           "## How this episode was made",
           "1. Pulled every item published in the last ~30 hours from the sources below.",
           "2. Gemini (editor pass) picked about 5 stories, assigned each to one section, and logged why the rest were skipped.",
           "3. Gemini (research pass) checked each picked story with Google Search and built a fact sheet from credible sources.",
           "4. Gemini (writer pass) wrote the two-host script from only those fact sheets.",
           "5. Gemini (fact-check pass) compared every claim in the script with the fact sheets and fixed or cut anything unsupported.",
           "6. Gemini voiced it; GitHub published it.", "",
           "## Sources checked"]
    out += [f"- {label}: {status}" for label, status in SOURCE_STATUS]
    out += ["", "## Used, by section"]
    for sec in plan.get("sections", []):
        out.append(f"### {sec.get('name', '')}")
        for st in sec.get("stories", []):
            it = by_id.get(st.get("id"), {})
            out.append(f"- {it.get('title', '?')} ({it.get('source', '')}) [link]({it.get('link', '')})  \n  Angle: {st.get('angle', '')}")
    out += ["", "## Fact sheets (verified with Google Search)"]
    for sec in plan.get("sections", []):
        for st in sec.get("stories", []):
            it = by_id.get(st.get("id"), {})
            out += ["", f"### {it.get('title', '?')}", it.get("facts") or "(No fact sheet: research failed, so only the original item text was used.)"]
            for t, u in it.get("fact_sources", [])[:8]:
                out.append(f"- Source: [{t}]({u})")
    out += ["", "## Fact-check fixes"] + ([f"- {f}" for f in fixes] or ["- none needed"])
    if plan.get("trends"):
        out += ["", "## Trends noted"] + [f"- {t}" for t in plan["trends"]]
    out += ["", "## Not used, and why"]
    for sk in plan.get("skipped", []):
        it = by_id.get(sk.get("id"), {})
        out.append(f"- {it.get('title', '?')} ({it.get('source', '')}): {sk.get('reason', '')}")
    d = DOCS / "logs"
    d.mkdir(exist_ok=True)
    (d / f"{slug}.md").write_text("\n".join(out) + "\n")
    return "\n".join(out)


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
    global VOICES
    try:
        return _tts_gemini(chunk, wav_path)
    except Exception as e:  # noqa: BLE001
        if VOICES == FALLBACK_VOICES:
            raise
        log(f"preferred voices failed ({e}); trying fallback voices")
        VOICES = dict(FALLBACK_VOICES)
        return _tts_gemini(chunk, wav_path)


# How the voices should say her name. The transcript keeps the real spelling; only the audio text changes.
NAME_SAY = "Chahnd-nee"


def say(text):
    return re.sub(r"\bChandni\b", NAME_SAY, text)


def _tts_gemini(chunk, wav_path):
    # Gemini 3.x TTS: every text part carries its own speaker tag; no untagged preamble allowed.
    body = {
        "contents": [{"role": "user", "parts": [
            {"text": say(t), "speechMetadata": {"speaker": s, "style": STYLES[s]}}
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
            r = subprocess.run([sys.executable, "-m", "edge_tts", "--voice", EDGE_VOICES[s], "--text", say(t),
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


def write_vtt(chunk_times, path, offset=0.0):
    """Timed transcript. Line timings inside each audio chunk are estimated from text length."""
    cues, t0 = ["WEBVTT", ""], offset
    for ch, dur in chunk_times:
        total = sum(len(t) for _, t in ch) or 1
        t = t0
        for speaker, text in ch:
            d = dur * len(text) / total
            cues += [f"{ts(t)} --> {ts(t + d)}", f"<v {speaker}>{text}", ""]
            t += d
        t0 += dur
    path.write_text("\n".join(cues))



MUSIC = Path(__file__).resolve().parent / "music"


def mix_music(voice, mp3_path):
    """Intro music, voices starting as it fades, outro music rising under the last line.
    Returns the second at which the voices start (to offset the transcript)."""
    intro, outro = MUSIC / "intro.mp3", MUSIC / "outro.mp3"
    if not (intro.exists() and outro.exists()):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(voice), "-af", "loudnorm=I=-16",
                        "-codec:a", "libmp3lame", "-b:a", "64k", str(mp3_path)], check=True)
        return 0.0
    lead = max(0.0, probe(intro) - 2.2)            # voices come in as the intro fades
    tail = max(0.0, probe(voice) + lead - 1.0)     # outro rises under the sign-off
    fc = (f"[0:a]aformat=sample_rates=44100:channel_layouts=mono,volume=0.55[i];"
          f"[1:a]aformat=sample_rates=44100:channel_layouts=mono,adelay={int(lead*1000)}[v];"
          f"[2:a]aformat=sample_rates=44100:channel_layouts=mono,volume=0.5,adelay={int(tail*1000)}[o];"
          f"[i][v][o]amix=inputs=3:normalize=0:duration=longest,loudnorm=I=-16[out]")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(intro), "-i", str(voice), "-i", str(outro),
                    "-filter_complex", fc, "-map", "[out]", "-codec:a", "libmp3lame", "-b:a", "64k", str(mp3_path)],
                   check=True)
    return lead


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
                        log(f"Gemini TTS failed ({e}); switching to free Edge voices for the WHOLE episode")
                        use_edge = True
                        # re-voice earlier chunks so the hosts don't change voice mid-episode
                        for j, (prev_ch, _) in enumerate(chunk_times):
                            tts_edge(prev_ch, wavs[j], tmp)
                            chunk_times[j] = (prev_ch, probe(wavs[j]))
                if use_edge:
                    tts_edge(ch, w, tmp)
            wavs.append(w)
            chunk_times.append((ch, probe(w)))
        joined = tmp / "all.wav"
        concat(wavs, joined)
        voice = tmp / "voice.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(joined), "-af", f"atempo={SPEED}",
                        "-ar", "44100", "-ac", "1", str(voice)], check=True)
        lead = mix_music(voice, mp3_path)
    chunk_times = [(ch, dur / SPEED) for ch, dur in chunk_times]
    write_vtt(chunk_times, mp3_path.with_suffix(".vtt"), offset=lead)
    return int(probe(mp3_path))


# ---------- main ----------
def main():
    if not KEY and not DRY:
        sys.exit("GEMINI_API_KEY is not set")
    now = dt.datetime.now(dt.timezone.utc)
    date_str = now.astimezone(dt.timezone(dt.timedelta(hours=-5))).strftime("%A, %B %-d, %Y")
    slug = now.strftime("%Y-%m-%d")
    EPISODES.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SKIP_IF_EXISTS") == "1" and (EPISODES / f"{slug}.mp3").exists():
        log(f"episode {slug} already exists; skipping backup run")
        return

    if not DRY:
        pick_models()
    items = gather_news()
    log(f"{len(items)} news items")
    if not items and not DRY:
        sys.exit("No news found; skipping today")
    plan = plan_episode(items)
    research_stories(items, plan)
    title, lines = write_script(items, plan, date_str)
    lines, fixes = fact_check(lines, items, plan)
    editor_log = write_editor_log(items, plan, slug, date_str, fixes)
    by_id = {it["id"]: it for it in items}
    save_covered(slug, [by_id[st["id"]]["title"] for sec in plan["sections"] for st in sec["stories"] if st["id"] in by_id])
    log(f"script: {len(lines)} lines, title: {title}")

    write_leads(extract_leads(items), slug, date_str)

    mp3 = EPISODES / f"{slug}.mp3"
    seconds = make_audio(lines, mp3)

    meta_path = DOCS / "episodes.json"
    eps = json.loads(meta_path.read_text()) if meta_path.exists() else []
    eps = [e for e in eps if e["slug"] != slug]
    eps.insert(0, {
        "slug": slug, "build": now.strftime("%H%M"), "title": title, "date": now.strftime("%a, %d %b %Y %H:%M:%S +0000"),
        "seconds": seconds, "bytes": mp3.stat().st_size,
        "notes": "Full transcript:\n\n" + "\n\n".join(f"{sp}: {t}" for sp, t in lines)
                 + "\n\n" + editor_log,
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
