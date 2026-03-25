"""SikhNet kirtan scraper with local STTM database lookups.

Discovers artists, fetches tracks with shabadId, enriches from database.sqlite,
filters/classifies/deduplicates, downloads audio, and pushes to HF Hub.

Usage:
    python src/data/sikhnet.py pilot          # 100-track validation
    python src/data/sikhnet.py build-catalog  # full artist discovery + enrichment
    python src/data/sikhnet.py download       # async MP3 download
    python src/data/sikhnet.py push-hf        # push to HF Hub in shards
    python src/data/sikhnet.py stats          # print catalog summary
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from functools import wraps
from pathlib import Path

import aiohttp
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

os.makedirs(config.LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOG_DIR, "scraper.log"), mode="a"),
    ],
)
log = logging.getLogger(__name__)

USER_AGENT = "GurbaniASR-DataPipeline/1.0"
SIKHNET_BASE = "https://play.sikhnet.com"
AUDIO_BASE = "https://www.sikhnet.com/gurbani/audio/play"

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "database.sqlite",
)

# ─── STTM ASCII → Unicode Gurmukhi ──────────────────────────────────────────
# Canonical mapping from @shabados/gurmukhi-utils unicode.jsonc

# Multi-character replacements (applied first, longest match first)
_ASCII_REPLACEMENTS = [
    ("<>", "\u0A74"),      # ੴ Ik Onkar
    ("<", "\u0A74"),       # ੴ
    (">", "\u262C"),       # ☬ Khanda
]

# Single-character mapping
_ASCII_MAP = {
    # Vowel carriers
    "a": "\u0A73",  # ੳ oora
    "A": "\u0A05",  # ਅ aira
    "e": "\u0A72",  # ੲ iri
    # Consonants
    "s": "\u0A38",  # ਸ
    "S": "\u0A36",  # ਸ਼ (sha)
    "h": "\u0A39",  # ਹ
    "H": "\u0A4D\u0A39",  # ੍ਹ pair ha
    "k": "\u0A15",  # ਕ
    "K": "\u0A16",  # ਖ
    "g": "\u0A17",  # ਗ
    "G": "\u0A18",  # ਘ
    "|": "\u0A19",  # ਙ
    "c": "\u0A1A",  # ਚ
    "C": "\u0A1B",  # ਛ
    "j": "\u0A1C",  # ਜ
    "J": "\u0A1D",  # ਝ
    "\\": "\u0A1E", # ਞ
    "t": "\u0A1F",  # ਟ
    "T": "\u0A20",  # ਠ
    "f": "\u0A21",  # ਡ
    "F": "\u0A22",  # ਢ
    "x": "\u0A23",  # ਣ
    "q": "\u0A24",  # ਤ
    "Q": "\u0A25",  # ਥ
    "d": "\u0A26",  # ਦ
    "D": "\u0A27",  # ਧ
    "n": "\u0A28",  # ਨ
    "p": "\u0A2A",  # ਪ
    "P": "\u0A2B",  # ਫ
    "b": "\u0A2C",  # ਬ
    "B": "\u0A2D",  # ਭ
    "m": "\u0A2E",  # ਮ
    "X": "\u0A2F",  # ਯ
    "r": "\u0A30",  # ਰ
    "l": "\u0A32",  # ਲ
    "L": "\u0A33",  # ਲ਼
    "v": "\u0A35",  # ਵ
    "V": "\u0A5C",  # ੜ
    # Vowel signs (matras)
    "w": "\u0A3E",  # ਾ kanna
    "W": "\u0A3E\u0A02",  # ਾਂ kanna+bindi
    "i": "\u0A3F",  # ਿ sihari
    "I": "\u0A40",  # ੀ bihari
    "u": "\u0A41",  # ੁ aunkar
    "U": "\u0A42",  # ੂ dulankar
    "y": "\u0A47",  # ੇ lavan
    "Y": "\u0A48",  # ੈ dulainkan
    "o": "\u0A4B",  # ੋ hora
    "O": "\u0A4C",  # ੌ kanaura
    "E": "\u0A13",  # ਓ
    # Nasals and modifiers
    "M": "\u0A70",  # ੰ tippi
    "N": "\u0A02",  # ਂ bindi
    "`": "\u0A71",  # ੱ adhak
    "~": "\u0A71",  # ੱ adhak (alt)
    "@": "\u0A51",  # ੑ udaat
    # Nukta forms
    "z": "\u0A5B",  # ਜ਼
    "Z": "\u0A5A",  # ਗ਼
    "^": "\u0A59",  # ਖ਼
    "&": "\u0A5E",  # ਫ਼
    # Subscript conjuncts
    "R": "\u0A4D\u0A30",  # ੍ਰ pair ra
    # Numbers
    "0": "\u0A66",  # ੦
    "1": "\u0A67",  # ੧
    "2": "\u0A68",  # ੨
    "3": "\u0A69",  # ੩
    "4": "\u0A6A",  # ੪
    "5": "\u0A6B",  # ੫
    "6": "\u0A6C",  # ੬
    "7": "\u0A6D",  # ੭
    "8": "\u0A6E",  # ੮
    "9": "\u0A6F",  # ੯
    # Punctuation
    "[": "\u0964",  # । single danda
    "]": "\u0965",  # ॥ double danda
    # Extended (from GurmukhiAkhar font)
    "\u00e6": "\u0A3C",  # ਼ nukta (æ)
    "\u00a1": "\u0A74",  # ੴ (¡)
    "\u0192": "\u0A28\u0A42\u0A70",  # ਨੂੰ (ƒ)
    "\u0153": "\u0A4D\u0A24",  # ੍ਤ (œ)
    "\u00cd": "\u0A4D\u0A35",  # ੍ਵ (Í)
    "\u00cf": "\u0A75",  # ੵ yakash (Ï)
    "\u00d2": "\u0965",  # ॥ (Ò)
    "\u00da": "\u0A03",  # ਃ visarga (Ú)
    "\u02c6": "\u0A02",  # ਂ bindi (ˆ)
    "\u02dc": "\u0A4D\u0A28",  # ੍ਨ (˜)
    "\u00a7": "\u0A4D\u0A39\u0A42",  # ੍ਹੂ (§)
    "\u00a4": "\u0A71",  # ੱ adhak (¤)
    "\u00e7": "\u0A4D\u0A1A",  # ੍ਚ (ç)
    "\u2020": "\u0A4D\u0A1F",  # ੍ਟ (†)
    "\u00fc": "\u0A41",  # ੁ (ü)
    "\u00ae": "\u0A4D\u0A30",  # ੍ਰ (®)
    "\u00b4": "\u0A75",  # ੵ yakash (´)
    "\u00a8": "\u0A42",  # ੂ (¨)
    "\u00b5": "\u0A70",  # ੰ tippi (µ)
}

# Characters nullified (produce empty string)
_ASCII_NULLIFY = set("\u00c6\u00d8\u00ff\u0152\u2030\u00d3\u00d4")

# Unicode sanitization: carrier+vowel → standalone vowel
_UNICODE_SANITIZE = [
    ("\u0A73\u0A4B", "\u0A13"),  # ੳੋ → ਓ
    ("\u0A05\u0A3E", "\u0A06"),  # ਅਾ → ਆ
    ("\u0A72\u0A3F", "\u0A07"),  # ੲਿ → ਇ
    ("\u0A72\u0A40", "\u0A08"),  # ੲੀ → ਈ
    ("\u0A73\u0A41", "\u0A09"),  # ੳੁ → ਉ
    ("\u0A73\u0A42", "\u0A0A"),  # ੳੂ → ਊ
    ("\u0A72\u0A47", "\u0A0F"),  # ੲੇ → ਏ
    ("\u0A05\u0A48", "\u0A10"),  # ਅੈ → ਐ
    ("\u0A05\u0A4C", "\u0A14"),  # ਅੌ → ਔ
]

# Gurmukhi base letters for sihari reordering
_BASE_LETTERS = set(
    "\u0A15\u0A16\u0A17\u0A18\u0A19"  # ka-nga
    "\u0A1A\u0A1B\u0A1C\u0A1D\u0A1E"  # cha-nya
    "\u0A1F\u0A20\u0A21\u0A22\u0A23"  # tta-nna
    "\u0A24\u0A25\u0A26\u0A27\u0A28"  # ta-na
    "\u0A2A\u0A2B\u0A2C\u0A2D\u0A2E"  # pa-ma
    "\u0A2F\u0A30\u0A32\u0A33\u0A35"  # ya-va
    "\u0A36\u0A38\u0A39"              # sha-ha
    "\u0A59\u0A5A\u0A5B\u0A5C\u0A5E"  # nukta forms
    "\u0A05\u0A06\u0A07\u0A08\u0A09\u0A0A"  # vowels
    "\u0A0F\u0A10\u0A13\u0A14"              # vowels
    "\u0A72\u0A73\u0A74"                     # carriers + ik onkar
)

SIHARI = "\u0A3F"  # ਿ


def ascii_to_unicode(text: str) -> str:
    """Convert STTM ASCII encoding to Unicode Gurmukhi.

    Handles the GurmukhiAkhar font-based encoding used in STTM database.
    Steps: multi-char replacements → single-char map → sihari reorder → sanitize.
    """
    if not text:
        return text

    # Step 1: Multi-character replacements (longest first)
    for ascii_seq, uni in _ASCII_REPLACEMENTS:
        text = text.replace(ascii_seq, uni)

    # Step 2: Single-character mapping
    result = []
    for ch in text:
        if ch in _ASCII_NULLIFY:
            continue
        result.append(_ASCII_MAP.get(ch, ch))
    text = "".join(result)

    # Step 3: Sihari reordering — move ਿ after the following base letter
    # In ASCII encoding, sihari appears BEFORE its consonant (visual order)
    # In Unicode, it must come AFTER (logical order)
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i] == SIHARI and chars[i + 1] in _BASE_LETTERS:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    text = "".join(chars)

    # Step 4: Unicode sanitization — carrier+vowel → standalone vowel
    for raw, clean in _UNICODE_SANITIZE:
        text = text.replace(raw, clean)

    # Step 5: Strip vishraam markers
    text = text.replace(";", "").replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Database lookups ────────────────────────────────────────────────────────

def lookup_shabad(sttm_id: int, db_path: str = DB_PATH) -> dict | None:
    """Look up shabad from local STTM database.sqlite by sttm_id.

    Returns dict with shabad_lines (Unicode), ang, raag, writer, or None.
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Get metadata
        cur.execute(
            """
            SELECT s.id, sec.name_english, w.name_english, l.source_page
            FROM shabads s
            JOIN lines l ON l.shabad_id = s.id
            JOIN sections sec ON s.section_id = sec.id
            JOIN writers w ON s.writer_id = w.id
            WHERE s.sttm_id = ?
            ORDER BY l.order_id
            LIMIT 1
            """,
            (sttm_id,),
        )
        meta = cur.fetchone()
        if not meta:
            return None

        shabad_db_id, raag, writer, ang = meta

        # Get all lines for this shabad
        cur.execute(
            "SELECT gurmukhi FROM lines WHERE shabad_id = ? ORDER BY order_id",
            (shabad_db_id,),
        )
        lines_ascii = [row[0] for row in cur.fetchall()]
        lines_unicode = [ascii_to_unicode(line) for line in lines_ascii]

        return {
            "shabad_lines": lines_unicode,
            "ang": ang,
            "raag": raag,
            "writer": writer,
        }
    finally:
        conn.close()


def gurmukhi_ratio(text: str) -> float:
    """Fraction of characters in Gurmukhi Unicode range U+0A00-U+0A7F."""
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    gurmukhi = sum(1 for c in chars if "\u0A00" <= c <= "\u0A7F")
    return gurmukhi / len(chars)


# ─── Retry decorator ─────────────────────────────────────────────────────────

def retry(max_retries=3, backoff_base=2):
    """Retry with exponential backoff on network/429 errors."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except requests.exceptions.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        wait = backoff_base ** (attempt + 1)
                        log.warning("429 rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                        time.sleep(wait)
                    elif attempt == max_retries:
                        raise
                    else:
                        wait = backoff_base ** attempt
                        log.warning("HTTP error %s, retrying in %ds", e, wait)
                        time.sleep(wait)
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                    if attempt == max_retries:
                        raise
                    wait = backoff_base ** attempt
                    log.warning("Network error %s, retrying in %ds", e, wait)
                    time.sleep(wait)
            return None
        return wrapper
    return decorator


# ─── SikhNet API ─────────────────────────────────────────────────────────────

_session = None

def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


@retry()
def discover_build_id() -> str:
    """Discover SikhNet Next.js buildId from page HTML."""
    resp = _get_session().get(f"{SIKHNET_BASE}/", timeout=15)
    resp.raise_for_status()
    # Look for /_next/data/{buildId}/ in script tags or __NEXT_DATA__
    match = re.search(r'"buildId"\s*:\s*"([^"]+)"', resp.text)
    if match:
        return match.group(1)
    # Fallback: look for _next/data/{id} in script src
    match = re.search(r'/_next/data/([^/]+)/', resp.text)
    if match:
        return match.group(1)
    raise RuntimeError("Could not discover SikhNet buildId from HTML")


_cached_build_id = None


def get_build_id() -> str:
    """Get cached buildId, re-discovering on 404."""
    global _cached_build_id
    if _cached_build_id is None:
        _cached_build_id = discover_build_id()
        log.info("Discovered buildId: %s", _cached_build_id)
    return _cached_build_id


def refresh_build_id() -> str:
    """Force re-discovery of buildId (on 404)."""
    global _cached_build_id
    _cached_build_id = None
    return get_build_id()


def _parse_playback_time(time_str: str) -> float:
    """Convert 'M:SS' or 'H:MM:SS' playback time string to seconds."""
    if not time_str:
        return 0.0
    parts = time_str.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return 0.0


def _extract_widget_items(data: dict) -> list[dict]:
    """Extract items from SikhNet Next.js pageProps response.

    Handles the nested structure: pageProps.pageData.standardPage.widgets[].rowsStackedWidget.items
    """
    page_props = data.get("pageProps", {})
    page_data = page_props.get("pageData", {})
    standard_page = page_data.get("standardPage", {})
    widgets = standard_page.get("widgets", [])

    all_items = []
    for widget in widgets:
        if not isinstance(widget, dict):
            continue
        for widget_key in ("rowsStackedWidget", "rowsCarouselWidget"):
            inner = widget.get(widget_key)
            if inner and isinstance(inner, dict):
                items = inner.get("items", [])
                all_items.extend(items)
    return all_items


def _track_info_to_dict(track_info: dict, artist_slug: str = "", from_playlist: str = "") -> dict:
    """Convert a SikhNet trackInfo object to our internal track dict."""
    duration = track_info.get("duration", 0)
    if isinstance(duration, str):
        duration = _parse_playback_time(duration)
    elif not duration:
        duration = _parse_playback_time(track_info.get("playbackTime", ""))

    # Extract artist slug from artistPathSlug (format: "artist/slug-name")
    a_slug = artist_slug
    a_path = track_info.get("artistPathSlug", "")
    if a_path and a_path.startswith("artist/"):
        a_slug = a_path[len("artist/"):]

    return {
        "track_id": track_info.get("id") or track_info.get("trackId"),
        "title": track_info.get("name", "") or track_info.get("title", ""),
        "artist_name": track_info.get("artistName", ""),
        "artist_id": track_info.get("artistId"),
        "artist_slug": a_slug,
        "duration_sec": duration,
        "shabad_id": track_info.get("shabadId"),
        "album": track_info.get("albumName", "") or track_info.get("albumTitle", ""),
        "url": track_info.get("resource", "") or track_info.get("url", ""),
        "_from_playlist": from_playlist,
    }


@retry()
def discover_artists(max_pages: int = 100) -> list[dict]:
    """Paginate SikhNet search API to discover unique artists.

    Extracts artists from both the "Artists" widget and trackInfo in the "Tracks" widget.
    Returns list of {slug, name, id} dicts.
    """
    seen = {}
    session = _get_session()
    search_terms = ["kirtan", "shabad", "gurbani", "raag"]

    for term in search_terms:
        for page in range(1, max_pages + 1):
            log.info("Discovering artists: q=%s page=%d...", term, page)
            resp = session.get(
                f"{SIKHNET_BASE}/api/search",
                params={"q": term, "type": "audio", "perPage": 50, "page": page},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            found_new = False
            for widget in data:
                rw = widget.get("rowsStackedWidget") or {}
                attrs = rw.get("attributes") or {}
                items = rw.get("items") or []
                title = attrs.get("title", "")

                for entry in items:
                    action = entry.get("action") or {}
                    path_slug = action.get("pathSlug", "") or ""
                    track_info = action.get("trackInfo")

                    # From "Artists" widget: pathSlug = "artist/slug-name"
                    if title == "Artists" and path_slug.startswith("artist/"):
                        slug = path_slug[len("artist/"):]
                        if slug and slug not in seen:
                            seen[slug] = {
                                "slug": slug,
                                "name": entry.get("title", ""),
                                "id": None,
                            }
                            found_new = True

                    # From "Tracks" widget: trackInfo.artistPathSlug
                    if track_info:
                        a_path = track_info.get("artistPathSlug", "")
                        if a_path and a_path.startswith("artist/"):
                            slug = a_path[len("artist/"):]
                            if slug and slug not in seen:
                                seen[slug] = {
                                    "slug": slug,
                                    "name": track_info.get("artistName", ""),
                                    "id": track_info.get("artistId"),
                                }
                                found_new = True

            time.sleep(config.DELAY_SIKHNET)

            # Stop paginating this term if no new artists found
            if not found_new:
                break

    artists = list(seen.values())
    log.info("Discovered %d unique artists", len(artists))
    return artists


@retry()
def get_artist_tracks(slug: str, build_id: str | None = None) -> list[dict]:
    """Fetch all tracks for an artist via SikhNet Next.js data API.

    Navigates: pageProps.pageData.standardPage.widgets[].rowsStackedWidget.items
    Each item has action.trackInfo with track details.
    """
    if build_id is None:
        build_id = get_build_id()

    session = _get_session()
    url = f"{SIKHNET_BASE}/_next/data/{build_id}/artist/{slug}/tracks.json"

    resp = session.get(url, timeout=15)
    if resp.status_code == 404:
        log.warning("404 on buildId %s, refreshing...", build_id)
        build_id = refresh_build_id()
        url = f"{SIKHNET_BASE}/_next/data/{build_id}/artist/{slug}/tracks.json"
        resp = session.get(url, timeout=15)

    resp.raise_for_status()
    data = resp.json()

    items = _extract_widget_items(data)
    tracks = []
    for item in items:
        track_info = (item.get("action") or {}).get("trackInfo")
        if not track_info:
            continue
        tracks.append(_track_info_to_dict(track_info, artist_slug=slug))

    return tracks


@retry()
def get_playlist_tracks(playlist_slug: str, build_id: str | None = None) -> list[dict]:
    """Fetch tracks from a SikhNet playlist."""
    if build_id is None:
        build_id = get_build_id()

    session = _get_session()
    url = f"{SIKHNET_BASE}/_next/data/{build_id}/playlist/{playlist_slug}.json"

    resp = session.get(url, timeout=15)
    if resp.status_code == 404:
        build_id = refresh_build_id()
        url = f"{SIKHNET_BASE}/_next/data/{build_id}/playlist/{playlist_slug}.json"
        resp = session.get(url, timeout=15)

    resp.raise_for_status()
    data = resp.json()

    items = _extract_widget_items(data)
    tracks = []
    for item in items:
        track_info = (item.get("action") or {}).get("trackInfo")
        if not track_info:
            continue
        tracks.append(_track_info_to_dict(track_info, from_playlist=playlist_slug))

    return tracks


def resolve_audio_url(track_id: int) -> str | None:
    """Resolve the direct audio URL by following the redirect."""
    try:
        resp = _get_session().head(
            f"{AUDIO_BASE}/{track_id}",
            allow_redirects=True,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.url
    except Exception as e:
        log.warning("Failed to resolve audio URL for track %s: %s", track_id, e)
    return None


# ─── Filtering ───────────────────────────────────────────────────────────────

BLACKLIST_KEYWORDS = [
    "akhand path", "akhand paath", "sehaj path", "sehaj paath", "sahaj path",
    "katha", "vichar", "vichaar", "teeka", "tika", "nitnem", "rehras sahib",
    "japji sahib", "sukhmani sahib", "asa di var", "lecture", "discourse",
    "dukh bhanjani", "ardas", "japji paath", "anand sahib paath",
]

BLACKLIST_ARTISTS = [
    "giani sant singh maskeen", "bhai pinderpal singh",
    "sant baba gurbachan singh", "khalsa prabandhak jatha",
    "bhai sarbjit singh dhunda",
]


def is_blacklisted(track: dict) -> bool:
    """Check if a track matches blacklist rules."""
    text = f"{track.get('title', '')} {track.get('album', '')}".lower()
    for kw in BLACKLIST_KEYWORDS:
        if kw in text:
            return True
    artist = track.get("artist_name", "").lower()
    for ba in BLACKLIST_ARTISTS:
        if ba in artist:
            return True
    return False


def passes_filter(track: dict, shabad_info: dict | None) -> bool:
    """Check if a track passes all filter rules."""
    # Must have shabadId
    shabad_id = track.get("shabad_id")
    if not shabad_id or shabad_id in (0, -1, "0", "-1"):
        return False

    # Duration filter
    dur = track.get("duration_sec", 0)
    if dur < config.MIN_DURATION_SEC or dur > config.MAX_DURATION_SEC:
        return False

    # Blacklist check
    if is_blacklisted(track):
        return False

    # Must have valid shabad in database
    if shabad_info is None:
        return False

    # Gurmukhi ratio check on shabad text
    all_text = " ".join(shabad_info.get("shabad_lines", []))
    if gurmukhi_ratio(all_text) < config.MIN_GURMUKHI_RATIO:
        return False

    return True


# ─── Style classification ───────────────────────────────────────────────────

HAZOORI_ARTISTS = [
    "bhai harjinder singh", "prof darshan singh", "bhai niranjan singh",
]
HAZOORI_KEYWORDS = ["darbar sahib", "harmandir sahib", "golden temple"]

PURATAN_ARTISTS = [
    "bhai surinder singh jodhpuri", "bhai avtar singh",
]
PURATAN_KEYWORDS = ["classical", "raag ", "puratan"]

AKJ_ARTISTS = ["akj", "bhai amolak singh"]
AKJ_KEYWORDS = ["akj", "rainsbhai", "akhand kirtani"]

TAKSALI_KEYWORDS = ["taksal", "taksali", "damdami"]
LIVE_KEYWORDS = ["gurdwara", "gurudwara", "diwan", "samagam"]


def classify_style(track: dict) -> str:
    """Classify track into a style_bucket."""
    text = f"{track.get('title', '')} {track.get('album', '')} {track.get('artist_name', '')}".lower()
    slug = track.get("artist_slug", "").lower()

    # AKJ (from playlist or keyword match)
    if track.get("_from_playlist") == "akhand-kirtan":
        return "akj"
    for kw in AKJ_KEYWORDS:
        if kw in text:
            return "akj"
    for a in AKJ_ARTISTS:
        if a in text:
            return "akj"

    # Hazoori
    for a in HAZOORI_ARTISTS:
        if a in text:
            return "hazoori"
    for kw in HAZOORI_KEYWORDS:
        if kw in text:
            return "hazoori"

    # Puratan
    for a in PURATAN_ARTISTS:
        if a in text:
            return "puratan"
    for kw in PURATAN_KEYWORDS:
        if kw in text:
            return "puratan"

    # Taksali
    for kw in TAKSALI_KEYWORDS:
        if kw in text:
            return "taksali"

    # Live
    for kw in LIVE_KEYWORDS:
        if kw in text:
            return "live"

    return "mixed"


def style_to_phase(style: str) -> int:
    """Map style_bucket to training phase."""
    return {"hazoori": 3, "puratan": 3}.get(style, 4)


# ─── Catalog building ───────────────────────────────────────────────────────

def make_recording_id(url: str) -> str:
    """Generate recording_id = md5(url)[:16]."""
    return hashlib.md5(url.encode()).hexdigest()[:16]


def build_catalog_entry(track: dict, shabad_info: dict, audio_url: str | None = None) -> dict:
    """Build a manifest entry from track + shabad info."""
    url = audio_url or track.get("url", "")
    recording_id = make_recording_id(url) if url else ""
    style = classify_style(track)

    return {
        "recording_id": recording_id,
        "url": url,
        "title": track.get("title", ""),
        "artist_name": track.get("artist_name", ""),
        "artist_id": track.get("artist_id"),
        "style_bucket": style,
        "source_name": "sikhnet",
        "sikhnet_track_id": track.get("track_id"),
        "shabad_id": track.get("shabad_id"),
        "shabad_lines": shabad_info["shabad_lines"],
        "ang": shabad_info["ang"],
        "raag": shabad_info["raag"],
        "writer": shabad_info["writer"],
        "duration_sec": track.get("duration_sec", 0),
        "local_path": f"{config.AUDIO_DIR}/{recording_id}.mp3" if recording_id else "",
        "alignment_status": "pending",
        "phase": style_to_phase(style),
    }


def dedup_catalog(entries: list[dict]) -> list[dict]:
    """Deduplicate by (shabad_id, artist_id) pair."""
    seen = set()
    deduped = []
    for entry in entries:
        key = (entry.get("shabad_id"), entry.get("artist_id"))
        if key not in seen:
            seen.add(key)
            deduped.append(entry)
    return deduped


# ─── Checkpoint / resume ────────────────────────────────────────────────────

def save_checkpoint(state: dict):
    """Save scraper checkpoint for resume."""
    os.makedirs(config.MANIFEST_DIR, exist_ok=True)
    with open(config.CHECKPOINT_FILE, "w") as f:
        json.dump(state, f, indent=2)


def load_checkpoint() -> dict:
    """Load scraper checkpoint if it exists."""
    if os.path.exists(config.CHECKPOINT_FILE):
        with open(config.CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"processed_artists": [], "catalog": []}


def save_catalog(catalog: list[dict]):
    """Save catalog to disk."""
    os.makedirs(config.MANIFEST_DIR, exist_ok=True)
    with open(config.CATALOG_FILE, "w") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    log.info("Saved catalog with %d entries to %s", len(catalog), config.CATALOG_FILE)


def load_catalog() -> list[dict]:
    """Load catalog from disk."""
    if os.path.exists(config.CATALOG_FILE):
        with open(config.CATALOG_FILE) as f:
            return json.load(f)
    return []


# ─── Async audio download ───────────────────────────────────────────────────

async def download_one(
    session: aiohttp.ClientSession,
    entry: dict,
    semaphore: asyncio.Semaphore,
) -> bool:
    """Download a single audio file."""
    url = entry.get("url")
    local_path = entry.get("local_path")
    if not url or not local_path:
        return False

    if os.path.exists(local_path):
        return True

    async with semaphore:
        try:
            async with session.get(url, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    log.warning("Download failed %s: HTTP %d", url, resp.status)
                    return False
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                with open(local_path, "wb") as f:
                    async for chunk in resp.content.iter_chunked(8192):
                        f.write(chunk)
            log.info("Downloaded %s", os.path.basename(local_path))
            await asyncio.sleep(config.DELAY_DOWNLOAD)
            return True
        except Exception as e:
            log.warning("Download error %s: %s", url, e)
            return False


async def download_all(entries: list[dict]) -> int:
    """Download all audio files with concurrency limit."""
    sem = asyncio.Semaphore(config.MAX_DL_CONCURRENT)
    headers = {"User-Agent": USER_AGENT}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = [download_one(session, e, sem) for e in entries]
        results = await asyncio.gather(*tasks)
    downloaded = sum(1 for r in results if r)
    log.info("Downloaded %d / %d files", downloaded, len(entries))
    return downloaded


# ─── CLI commands ────────────────────────────────────────────────────────────

def cmd_pilot(limit: int = 100):
    """Run pilot: discover a few artists, collect ~100 valid tracks."""
    log.info("=== PILOT: targeting %d tracks ===", limit)

    build_id = get_build_id()
    catalog = []

    # Start with AKJ playlist (high shabadId rate)
    log.info("Fetching AKJ playlist...")
    try:
        akj_tracks = get_playlist_tracks("akhand-kirtan", build_id)
        log.info("AKJ playlist: %d tracks", len(akj_tracks))
    except Exception as e:
        log.warning("AKJ playlist failed: %s", e)
        akj_tracks = []

    # Discover some artists
    log.info("Discovering artists (first 5 pages)...")
    artists = discover_artists(max_pages=5)

    # Process AKJ tracks first
    all_tracks = akj_tracks
    for artist in artists[:20]:  # limit to 20 artists for pilot
        log.info("Fetching tracks for %s...", artist["name"])
        try:
            tracks = get_artist_tracks(artist["slug"], build_id)
            all_tracks.extend(tracks)
            time.sleep(config.DELAY_SIKHNET)
        except Exception as e:
            log.warning("Failed to get tracks for %s: %s", artist["slug"], e)

    log.info("Total raw tracks: %d", len(all_tracks))

    # Filter and enrich
    for track in all_tracks:
        if len(catalog) >= limit:
            break

        shabad_id = track.get("shabad_id")
        if not shabad_id or shabad_id in (0, -1, "0", "-1"):
            continue

        shabad_info = lookup_shabad(int(shabad_id))
        if not passes_filter(track, shabad_info):
            continue

        # Resolve audio URL if not present
        audio_url = track.get("url")
        if not audio_url and track.get("track_id"):
            audio_url = resolve_audio_url(track["track_id"])

        entry = build_catalog_entry(track, shabad_info, audio_url)
        catalog.append(entry)

    catalog = dedup_catalog(catalog)
    save_catalog(catalog)

    log.info("=== PILOT COMPLETE: %d valid tracks ===", len(catalog))
    _print_stats(catalog)
    return catalog


def cmd_build_catalog():
    """Full artist discovery and catalog building with checkpoint/resume."""
    log.info("=== BUILD CATALOG ===")

    checkpoint = load_checkpoint()
    processed_slugs = set(checkpoint.get("processed_artists", []))
    catalog = checkpoint.get("catalog", [])

    build_id = get_build_id()

    # AKJ playlist
    if "__akj__" not in processed_slugs:
        log.info("Fetching AKJ playlist...")
        try:
            akj_tracks = get_playlist_tracks("akhand-kirtan", build_id)
            for track in akj_tracks:
                shabad_id = track.get("shabad_id")
                if not shabad_id or shabad_id in (0, -1):
                    continue
                shabad_info = lookup_shabad(int(shabad_id))
                if passes_filter(track, shabad_info):
                    audio_url = track.get("url") or resolve_audio_url(track["track_id"])
                    catalog.append(build_catalog_entry(track, shabad_info, audio_url))
            processed_slugs.add("__akj__")
        except Exception as e:
            log.error("AKJ playlist error: %s", e)

    # Discover all artists
    artists = discover_artists(max_pages=200)

    for i, artist in enumerate(artists):
        if artist["slug"] in processed_slugs:
            log.info("Skipping already-processed artist: %s", artist["name"])
            continue

        log.info("[%d/%d] Processing artist: %s", i + 1, len(artists), artist["name"])
        try:
            tracks = get_artist_tracks(artist["slug"], build_id)
            for track in tracks:
                shabad_id = track.get("shabad_id")
                if not shabad_id or shabad_id in (0, -1):
                    continue
                shabad_info = lookup_shabad(int(shabad_id))
                if passes_filter(track, shabad_info):
                    audio_url = track.get("url") or resolve_audio_url(track["track_id"])
                    catalog.append(build_catalog_entry(track, shabad_info, audio_url))
            time.sleep(config.DELAY_SIKHNET)
        except Exception as e:
            log.warning("Error processing artist %s: %s", artist["name"], e)

        processed_slugs.add(artist["slug"])

        # Checkpoint every 10 artists
        if (i + 1) % 10 == 0:
            catalog = dedup_catalog(catalog)
            save_checkpoint({
                "processed_artists": list(processed_slugs),
                "catalog": catalog,
            })
            log.info("Checkpoint saved: %d artists, %d entries", len(processed_slugs), len(catalog))

    catalog = dedup_catalog(catalog)
    save_catalog(catalog)
    save_checkpoint({"processed_artists": list(processed_slugs), "catalog": catalog})

    log.info("=== CATALOG COMPLETE: %d entries ===", len(catalog))
    _print_stats(catalog)
    return catalog


def cmd_download():
    """Download all audio files in the catalog."""
    catalog = load_catalog()
    if not catalog:
        log.error("No catalog found. Run build-catalog first.")
        return

    pending = [e for e in catalog if not os.path.exists(e.get("local_path", ""))]
    log.info("Downloading %d files (%d already exist)", len(pending), len(catalog) - len(pending))
    asyncio.run(download_all(pending))


def cmd_push_hf():
    """Push catalog + audio to HF Hub in shards."""
    try:
        from datasets import Audio, Dataset
        from huggingface_hub import HfApi
    except ImportError:
        log.error("Install datasets and huggingface_hub: pip install datasets huggingface_hub")
        return

    catalog = load_catalog()
    if not catalog:
        log.error("No catalog found.")
        return

    # Filter to entries with downloaded audio
    ready = [e for e in catalog if os.path.exists(e.get("local_path", ""))]
    log.info("Pushing %d entries to HF Hub (of %d total)", len(ready), len(catalog))

    shard_size = 500
    api = HfApi()

    for shard_idx in range(0, len(ready), shard_size):
        shard = ready[shard_idx : shard_idx + shard_size]
        shard_num = shard_idx // shard_size

        ds_dict = {
            "audio": [e["local_path"] for e in shard],
            "text": [" ".join(e["shabad_lines"]) for e in shard],
            "recording_id": [e["recording_id"] for e in shard],
            "shabad_id": [e["shabad_id"] for e in shard],
            "ang": [e["ang"] for e in shard],
            "raag": [e["raag"] for e in shard],
            "writer": [e["writer"] for e in shard],
            "style_bucket": [e["style_bucket"] for e in shard],
            "phase": [e["phase"] for e in shard],
            "duration_sec": [e["duration_sec"] for e in shard],
            "artist_name": [e["artist_name"] for e in shard],
            "title": [e["title"] for e in shard],
        }

        ds = Dataset.from_dict(ds_dict)
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        ds.push_to_hub(
            config.HF_DATASET_REPO,
            split=f"train_shard_{shard_num:04d}",
            token=os.environ.get("HF_TOKEN"),
        )
        log.info("Pushed shard %d (%d entries)", shard_num, len(shard))


def cmd_stats():
    """Print catalog statistics."""
    catalog = load_catalog()
    if not catalog:
        print("No catalog found.")
        return
    _print_stats(catalog)


def _print_stats(catalog: list[dict]):
    """Print summary statistics for a catalog."""
    total = len(catalog)
    downloaded = sum(1 for e in catalog if os.path.exists(e.get("local_path", "")))
    total_hours = sum(e.get("duration_sec", 0) for e in catalog) / 3600

    # Style distribution
    styles = {}
    for e in catalog:
        s = e.get("style_bucket", "unknown")
        styles[s] = styles.get(s, 0) + 1

    # Phase distribution
    phases = {}
    for e in catalog:
        p = e.get("phase", 0)
        phases[p] = phases.get(p, 0) + 1

    # Raag distribution (top 10)
    raags = {}
    for e in catalog:
        r = e.get("raag", "unknown")
        raags[r] = raags.get(r, 0) + 1

    print(f"\n{'='*50}")
    print(f"  Catalog: {total} entries")
    print(f"  Downloaded: {downloaded} / {total}")
    print(f"  Total duration: {total_hours:.1f} hours")
    print(f"\n  Style buckets:")
    for s, c in sorted(styles.items(), key=lambda x: -x[1]):
        print(f"    {s:12s}: {c:5d}")
    print(f"\n  Training phases:")
    for p, c in sorted(phases.items()):
        print(f"    Phase {p}: {c:5d}")
    print(f"\n  Top raags:")
    for r, c in sorted(raags.items(), key=lambda x: -x[1])[:10]:
        print(f"    {r:30s}: {c:5d}")
    print(f"{'='*50}\n")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    # Ensure directories exist
    os.makedirs(config.AUDIO_DIR, exist_ok=True)
    os.makedirs(config.MANIFEST_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "pilot":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        cmd_pilot(limit)
    elif cmd == "build-catalog":
        cmd_build_catalog()
    elif cmd == "download":
        cmd_download()
    elif cmd == "push-hf":
        cmd_push_hf()
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
