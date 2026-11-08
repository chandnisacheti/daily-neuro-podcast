# Neuro Morning

A free, fully automatic daily podcast on neurotech, BCIs, wearables, healthtech and medtech.

Every morning a GitHub Action gathers the last 24 hours of news (RSS feeds in `podcast/sources.py`),
Gemini writes a ~10 minute two-host script and voices it (free tier), and the MP3 plus `feed.xml`
are published to GitHub Pages from `docs/`.

## One-time setup
1. Repo secret `GEMINI_API_KEY` (free key from https://aistudio.google.com/apikey).
2. Settings → Pages → Source: "Deploy from a branch", branch `main`, folder `/docs`.
3. Actions tab → "Daily episode" → Run workflow (first test).
4. In Apple Podcasts: Library → ⋯ → Follow a Show by URL →
   `https://<your-username>.github.io/daily-neuro-podcast/feed.xml`

## Tweaks
- Time: edit the `cron` line in `.github/workflows/daily.yml` (it is in UTC).
- Sources: edit `podcast/sources.py`.
- Hosts, tone, length: edit `PROMPT` in `podcast/make_episode.py`.
- If Gemini's free voices fail, it falls back to free Microsoft Edge voices automatically.
