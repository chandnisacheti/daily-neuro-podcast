"""News sources for the daily episode. Add or remove feeds freely; broken feeds are skipped."""
from urllib.parse import quote_plus


def google_news(query: str) -> str:
    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(f"{query} when:1d")
        + "&hl=en-US&gl=US&ceid=US:en"
    )


# (label, url, max items taken from this feed)
FEEDS = [
    ("Neurotechnology Substack", "https://neurotechnology.substack.com/feed", 5),
    ("Neuroscience News", "https://neurosciencenews.com/feed/", 8),
    ("MedTech Dive", "https://www.medtechdive.com/feeds/news/", 10),
    ("Fierce Biotech MedTech", "https://www.fiercebiotech.com/rss/medtech/xml", 8),
    ("STAT Health Tech", "https://www.statnews.com/category/health-tech/feed/", 8),
    ("MobiHealthNews", "https://www.mobihealthnews.com/feed", 8),
    ("TechCrunch Health", "https://techcrunch.com/tag/health/feed/", 6),
    ("FDA Medical Devices", "https://www.fda.gov/about-fda/contact-fda/stay-informed/rss-feeds/medical-devices/rss.xml", 6),
    ("Google News: neurotech", google_news('neurotech OR neurotechnology OR "brain-computer interface" OR BCI'), 12),
    ("Google News: neuro funding", google_news('(neurotech OR "brain-computer" OR neuromodulation OR "digital health" OR wearable) (raises OR funding OR "Series A" OR "Series B" OR seed)'), 10),
    ("Google News: FDA neuro devices", google_news('FDA (clearance OR approval OR "breakthrough device") (neuro OR brain OR wearable OR implant)'), 8),
    ("Google News: NYC neurotech", google_news('("New York" OR NYC OR Brooklyn) (neurotech OR "brain-computer" OR medtech OR "health tech" startup)'), 8),
    ("Google News: wearables", google_news('wearable (EEG OR biosensor OR "health monitoring" OR sleep OR HRV)'), 8),
    ("bioRxiv Neuroscience", "https://connect.biorxiv.org/biorxiv_xml/subject/neuroscience", 6),
]
