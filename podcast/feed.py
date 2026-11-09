"""Write docs/feed.xml, a podcast RSS feed Apple Podcasts / Overcast / Pocket Casts can follow."""
import os
from pathlib import Path
from xml.sax.saxutils import escape


def site_url():
    if os.environ.get("SITE_URL"):
        return os.environ["SITE_URL"].rstrip("/")
    repo = os.environ.get("GITHUB_REPOSITORY", "owner/daily-neuro-podcast")
    owner, name = repo.split("/")
    return f"https://{owner.lower()}.github.io/{name}"


def write_feed(episodes, docs: Path):
    base = site_url()
    items = []
    for e in episodes:
        # A same-day re-run gets a new build id, so podcast apps (Spotify caches audio by guid) fetch the new file.
        b = e.get("build", "")
        url = f"{base}/episodes/{e['slug']}.mp3" + (f"?v={b}" if b else "")
        vtt = docs / "episodes" / f"{e['slug']}.vtt"
        transcript = (f'\n   <podcast:transcript url="{base}/episodes/{e["slug"]}.vtt" type="text/vtt" language="en"/>'
                      if vtt.exists() else "")
        items.append(f"""  <item>
   <title>{escape(e['title'])}</title>
   <description>{escape(e['notes'])}</description>
   <pubDate>{e['date']}</pubDate>
   <guid isPermaLink="false">neuro-morning-{e['slug']}{'-' + b if b else ''}</guid>
   <enclosure url="{url}" length="{e['bytes']}" type="audio/mpeg"/>
   <itunes:duration>{e['seconds']}</itunes:duration>
   <itunes:explicit>false</itunes:explicit>{transcript}
  </item>""")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd" xmlns:podcast="https://podcastindex.org/namespace/1.0">
 <channel>
  <title>Neuro Morning</title>
  <link>{base}/</link>
  <language>en-us</language>
  <description>A private daily AI briefing on neurotech, BCIs, wearables, healthtech and medtech.</description>
  <itunes:author>Neuro Morning</itunes:author>
  <itunes:owner><itunes:name>Chandni Sacheti</itunes:name><itunes:email>chandnisacheti@gmail.com</itunes:email></itunes:owner>
  <podcast:locked>yes</podcast:locked>
  <itunes:image href="{base}/cover-sunrise.jpg"/>
  <itunes:explicit>false</itunes:explicit>
  <itunes:block>Yes</itunes:block>
  <itunes:category text="Science"/>
{chr(10).join(items)}
 </channel>
</rss>
"""
    (docs / "feed.xml").write_text(xml)
