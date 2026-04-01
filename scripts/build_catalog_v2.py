#!/usr/bin/env python3
"""Build test_catalog_v2.json: 30 SikhNet tracks across 6 style buckets (5 each).

Queries SikhNet search API for tracks, classifies by style, looks up
shabad_lines from database.sqlite, outputs test_catalog_v2.json.

Usage:
    python3 scripts/build_catalog_v2.py
    python3 scripts/build_catalog_v2.py --dry-run   # print catalog, don't save
"""

import argparse
import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "database.sqlite"
OUTPUT = "data/manifests/test_catalog_v2.json"
TARGET_PER_BUCKET = 5
MIN_DUR_SEC = 120   # 2 min
MAX_DUR_SEC = 5400  # 90 min

# ─── Style bucket classification ─────────────────────────────────────────────

BUCKET_RULES = [
    # (bucket, keywords_in_artist_or_title)
    ("hazoori", ["harjinder singh", "darshan singh", "darbar sahib", "hazoori",
                 "srinagar wale", "bhai guriqbal"]),
    ("puratan", ["jodhpuri", "jodhpur", "surinder singh", "puratan", "raag bhairav",
                 "raag todi", "bhai avtar singh"]),
    ("akj",     ["akj", "rainsbhai", "akhand kirtan", "amolak singh",
                 "bhai jasbir singh", "bhai harpreet singh akj"]),
    ("taksali", ["taksal", "taksali", "damdami", "bhai satnam singh sethi",
                 "bhai jagjit singh"]),
    ("live",    ["gurdwara", "gurudwara", "diwan", "live kirtan",
                 "samagam", "smagam"]),
]


def classify_bucket(artist: str, title: str, album: str = "") -> str:
    text = f"{artist} {title} {album}".lower()
    for bucket, keywords in BUCKET_RULES:
        if any(k in text for k in keywords):
            return bucket
    return "mixed"


# ─── STTM ASCII → Unicode ─────────────────────────────────────────────────────

_ASCII_REPLACEMENTS = [("<>", "\u0A74"), ("<", "\u0A74"), (">", "\u262C")]
_ASCII_MAP = {
    "a": "\u0A73", "A": "\u0A05", "e": "\u0A72",
    "s": "\u0A38", "S": "\u0A36", "h": "\u0A39", "H": "\u0A4D\u0A39",
    "k": "\u0A15", "K": "\u0A16", "g": "\u0A17", "G": "\u0A18", "|": "\u0A19",
    "c": "\u0A1A", "C": "\u0A1B", "j": "\u0A1C", "J": "\u0A1D", "\\": "\u0A1E",
    "t": "\u0A1F", "T": "\u0A20", "f": "\u0A21", "F": "\u0A22", "x": "\u0A23",
    "q": "\u0A24", "Q": "\u0A25", "d": "\u0A26", "D": "\u0A27", "n": "\u0A28",
    "p": "\u0A2A", "P": "\u0A2B", "b": "\u0A2C", "B": "\u0A2D", "m": "\u0A2E",
    "X": "\u0A2F", "r": "\u0A30", "l": "\u0A32", "L": "\u0A33", "v": "\u0A35",
    "V": "\u0A5C",
    "w": "\u0A3E", "W": "\u0A3E\u0A02", "i": "\u0A3F", "I": "\u0A40",
    "u": "\u0A41", "U": "\u0A42", "y": "\u0A47", "Y": "\u0A48",
    "o": "\u0A4B", "O": "\u0A4C", "E": "\u0A13",
    "M": "\u0A70", "N": "\u0A02", "`": "\u0A71", "~": "\u0A71", "@": "\u0A51",
    "z": "\u0A5B", "Z": "\u0A5A", "^": "\u0A59", "&": "\u0A5E",
    "R": "\u0A4D\u0A30",
    "0": "\u0A66", "1": "\u0A67", "2": "\u0A68", "3": "\u0A69", "4": "\u0A6A",
    "5": "\u0A6B", "6": "\u0A6C", "7": "\u0A6D", "8": "\u0A6E", "9": "\u0A6F",
    "[": "\u0964", "]": "\u0965",
    "\u00e6": "\u0A3C", "\u00a1": "\u0A74",
    "\u0192": "\u0A28\u0A42\u0A70", "\u0153": "\u0A4D\u0A24",
    "\u00cd": "\u0A4D\u0A35", "\u00cf": "\u0A75", "\u00d2": "\u0965",
    "\u00da": "\u0A03", "\u02c6": "\u0A02", "\u02dc": "\u0A4D\u0A28",
    "\u00a7": "\u0A4D\u0A39\u0A42", "\u00a4": "\u0A71",
    "\u00e7": "\u0A4D\u0A1A", "\u2020": "\u0A4D\u0A1F",
    "\u00fc": "\u0A41", "\u00ae": "\u0A4D\u0A30",
    "\u00b4": "\u0A75", "\u00a8": "\u0A42", "\u00b5": "\u0A70",
}
_ASCII_NULLIFY = set("\u00c6\u00d8\u00ff\u0152\u2030\u00d3\u00d4")
_UNICODE_SANITIZE = [
    ("\u0A73\u0A4B", "\u0A13"), ("\u0A05\u0A3E", "\u0A06"),
    ("\u0A72\u0A3F", "\u0A07"), ("\u0A72\u0A40", "\u0A08"),
    ("\u0A73\u0A41", "\u0A09"), ("\u0A73\u0A42", "\u0A0A"),
    ("\u0A72\u0A47", "\u0A0F"), ("\u0A05\u0A48", "\u0A10"),
    ("\u0A05\u0A4C", "\u0A14"),
]
_BASE_LETTERS = set(
    "\u0A15\u0A16\u0A17\u0A18\u0A19\u0A1A\u0A1B\u0A1C\u0A1D\u0A1E"
    "\u0A1F\u0A20\u0A21\u0A22\u0A23\u0A24\u0A25\u0A26\u0A27\u0A28"
    "\u0A2A\u0A2B\u0A2C\u0A2D\u0A2E\u0A2F\u0A30\u0A32\u0A33\u0A35"
    "\u0A36\u0A38\u0A39\u0A59\u0A5A\u0A5B\u0A5C\u0A5E"
    "\u0A05\u0A06\u0A07\u0A08\u0A09\u0A0A\u0A0F\u0A10\u0A13\u0A14"
    "\u0A72\u0A73\u0A74"
)
SIHARI = "\u0A3F"


def ascii_to_unicode(text: str) -> str:
    if not text:
        return text
    for ascii_seq, uni in _ASCII_REPLACEMENTS:
        text = text.replace(ascii_seq, uni)
    result = []
    for ch in text:
        if ch in _ASCII_NULLIFY:
            continue
        result.append(_ASCII_MAP.get(ch, ch))
    text = "".join(result)
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if chars[i] == SIHARI and chars[i + 1] in _BASE_LETTERS:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        else:
            i += 1
    text = "".join(chars)
    for raw, clean in _UNICODE_SANITIZE:
        text = text.replace(raw, clean)
    text = text.replace(";", "").replace(".", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Database lookup ──────────────────────────────────────────────────────────

def lookup_shabad(sttm_id: int, db_path: str = DB_PATH) -> dict | None:
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT s.id, sec.name_english, w.name_english, l.source_page
            FROM shabads s
            JOIN lines l ON l.shabad_id = s.id
            JOIN sections sec ON s.section_id = sec.id
            JOIN writers w ON s.writer_id = w.id
            WHERE s.sttm_id = ? ORDER BY l.order_id LIMIT 1""",
            (sttm_id,),
        )
        meta = cur.fetchone()
        if not meta:
            return None
        shabad_db_id, raag, writer, ang = meta
        cur.execute(
            "SELECT gurmukhi FROM lines WHERE shabad_id = ? ORDER BY order_id",
            (shabad_db_id,),
        )
        lines_ascii = [row[0] for row in cur.fetchall()]
        lines_unicode = [ascii_to_unicode(line) for line in lines_ascii if line.strip()]
        if not lines_unicode:
            return None
        return {"shabad_lines": lines_unicode, "ang": ang, "raag": raag, "writer": writer}
    finally:
        conn.close()


# ─── SikhNet API ─────────────────────────────────────────────────────────────

def discover_build_id() -> str:
    """Get current SikhNet buildId from any artist page."""
    resp = requests.get(
        "https://play.sikhnet.com/artist/bhai-davinder-singh-sodhi-ludhiana",
        headers={"User-Agent": "Mozilla/5.0 GurbaniASR/1.0"},
        timeout=20,
    )
    m = re.search(r'"buildId":"([^"]+)"', resp.text)
    if m:
        return m.group(1)
    raise RuntimeError("Could not discover SikhNet buildId")


def get_artist_tracks(slug: str, build_id: str) -> list[dict]:
    """Fetch all tracks for an artist via SikhNet _next/data API."""
    url = f"https://play.sikhnet.com/_next/data/{build_id}/artist/{slug}/tracks.json"
    try:
        resp = requests.get(url, headers={"User-Agent": "GurbaniASR/1.0"}, timeout=20)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"    Artist {slug}: {e}")
        return []

    pages = data.get("pageProps", {})
    std = pages.get("pageData", {}).get("standardPage", {})
    widgets = std.get("widgets", [])
    tracks = []
    for widget in widgets:
        items = widget.get("rowsStackedWidget", {}).get("items", []) or []
        for item in items:
            action = item.get("action") or {}
            ti = action.get("trackInfo") or {}
            resource = action.get("resource", "")
            shabad_id = ti.get("shabadId")
            if not shabad_id or not resource:
                continue
            m2 = re.search(r"/play/(\d+)", resource)
            if not m2:
                continue
            artist_name = ti.get("artistName", "") or item.get("subtitle", "")
            title = ti.get("title", "") or item.get("title", "")
            playback = ti.get("playbackTime", "") or item.get("rightTitle", "")
            tracks.append({
                "sikhnet_track_id": int(m2.group(1)),
                "url": resource,
                "title": title,
                "artist_name": artist_name,
                "artist_id": ti.get("artistId", 0),
                "shabad_id": shabad_id,
                "duration_sec": parse_duration(str(playback)),
                "album": item.get("supertitle", ""),
                "artist_path_slug": ti.get("artistPathSlug", f"artist/{slug}"),
            })
    return tracks


# Known artist slugs per style bucket (SikhNet URL slugs)
BUCKET_ARTIST_SLUGS = {
    "hazoori": [
        "bhai-harjinder-singh-srinagar-wale",
        "bhai-balbir-singh",
        "prof-darshan-singh",
        "bhai-guriqbal-singh",
        "bhai-nirmal-singh-khalsa",
    ],
    "puratan": [
        "bhai-surinder-singh-jodhpuri",
        "bhai-avtar-singh",
        "bhai-gurcharan-singh-ragi",
        "bhai-sant-singh",
    ],
    "akj": [
        "bhai-amolak-singh",
        "bhai-jaspal-singh-akj",
        "bhai-kulwant-singh-akj",
    ],
    "taksali": [
        "bhai-jagjit-singh",
        "bhai-satnam-singh-sethi",
        "bhai-paramjit-singh-khalsa",
    ],
    "live": [
        "bhai-sarabjit-singh-randhawa",
        "bhai-amarjit-singh-taan",
    ],
    "mixed": [
        "bhai-davinder-singh-sodhi-ludhiana",
        "bhai-dilbagh-singh-gulbagh-singh",
    ],
}


# ─── Fallback: collect slugs from search API ─────────────────────────────────

SEARCH_QUERIES = ["kirtan", "gurbani", "shabad", "raag", "taksal",
                  "gurdwara", "diwan", "classical", "rainsbhai"]

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


def is_blacklisted(title: str, artist: str) -> bool:
    text = f"{title} {artist}".lower()
    if any(k in text for k in BLACKLIST_KEYWORDS):
        return True
    if any(k in artist.lower() for k in BLACKLIST_ARTISTS):
        return True
    return False


def parse_duration(playback_time: str) -> float:
    """Convert 'M:SS' or 'H:MM:SS' string to seconds."""
    if not playback_time:
        return 0.0
    parts = playback_time.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except (ValueError, IndexError):
        pass
    return 0.0


def search_for_artist_slugs(build_id: str) -> dict[str, list[str]]:
    """Fallback: discover artist slugs from search API, classify by bucket."""
    discovered: dict[str, set[str]] = {b: set() for b in BUCKET_ARTIST_SLUGS}
    for query in SEARCH_QUERIES:
        try:
            resp = requests.get(
                "https://play.sikhnet.com/api/search",
                params={"q": query, "type": "audio", "perPage": 50, "page": 1},
                headers={"User-Agent": "GurbaniASR/1.0"}, timeout=20,
            )
            data = resp.json()
            for widget in data:
                for item in (widget.get("rowsStackedWidget", {}).get("items") or []):
                    ti = (item.get("action") or {}).get("trackInfo") or {}
                    slug_full = ti.get("artistPathSlug", "")  # "artist/bhai-..."
                    if not slug_full:
                        continue
                    slug = slug_full.replace("artist/", "")
                    artist = ti.get("artistName", "")
                    title = ti.get("title", "")
                    bucket = classify_bucket(artist, title)
                    discovered[bucket].add(slug)
            time.sleep(0.3)
        except Exception:
            pass
    return {b: list(v) for b, v in discovered.items()}


# ─── Catalog builder ─────────────────────────────────────────────────────────

def build_catalog(dry_run: bool = False) -> list[dict]:
    seen_shabad_artists: set[tuple] = set()
    buckets: dict[str, list] = {b: [] for b in BUCKET_ARTIST_SLUGS}

    print("Discovering SikhNet buildId...")
    try:
        build_id = discover_build_id()
        print(f"  buildId: {build_id}")
    except Exception as e:
        print(f"  ERROR: {e}")
        return []

    # Build slug list: start with known slugs, supplement from search
    slug_map = {b: list(slugs) for b, slugs in BUCKET_ARTIST_SLUGS.items()}
    print("Supplementing with search-discovered slugs...")
    extra = search_for_artist_slugs(build_id)
    for bucket, slugs in extra.items():
        for s in slugs:
            if s not in slug_map[bucket]:
                slug_map[bucket].append(s)

    # Fetch tracks per bucket
    print("\nFetching artist tracks per bucket...")
    all_candidates: list[dict] = []
    for bucket, slugs in slug_map.items():
        for slug in slugs:
            if len(buckets[bucket]) >= TARGET_PER_BUCKET:
                break
            print(f"  [{bucket}] artist/{slug}")
            tracks = get_artist_tracks(slug, build_id)
            print(f"    → {len(tracks)} tracks")
            for t in tracks:
                t["_bucket_hint"] = bucket
            all_candidates.extend(tracks)
            time.sleep(0.5)

    print(f"\nTotal raw candidates: {len(all_candidates)}")

    # Process and classify
    for cand in all_candidates:
        shabad_id = cand["shabad_id"]
        artist_id = cand.get("artist_id", 0)
        dedup_key = (shabad_id, artist_id)
        if dedup_key in seen_shabad_artists:
            continue

        title = cand["title"]
        artist = cand["artist_name"]

        if is_blacklisted(title, artist):
            continue

        dur = cand["duration_sec"]
        if dur > 0 and (dur < MIN_DUR_SEC or dur > MAX_DUR_SEC):
            continue

        # Use bucket_hint from artist page fetch, then fall back to classify
        bucket = cand.get("_bucket_hint") or classify_bucket(artist, title, cand.get("album", ""))
        if len(buckets[bucket]) >= TARGET_PER_BUCKET:
            continue

        # Look up shabad from database.sqlite
        shabad_data = lookup_shabad(shabad_id)
        if not shabad_data or len(shabad_data["shabad_lines"]) < 2:
            continue

        seen_shabad_artists.add(dedup_key)
        url = cand["url"]
        recording_id = hashlib.md5(url.encode()).hexdigest()[:16]

        phase = 3 if bucket in ("hazoori", "puratan") else 4

        entry = {
            "recording_id": recording_id,
            "url": url,
            "title": title,
            "artist_name": artist,
            "artist_id": artist_id,
            "style_bucket": bucket,
            "source_name": "sikhnet",
            "sikhnet_track_id": cand["sikhnet_track_id"],
            "shabad_id": shabad_id,
            "shabad_lines": shabad_data["shabad_lines"],
            "ang": shabad_data["ang"],
            "raag": shabad_data["raag"],
            "writer": shabad_data["writer"],
            "duration_sec": dur,
            "local_path": f"data/audio/{recording_id}.mp3",
            "alignment_status": "pending",
            "phase": phase,
        }
        buckets[bucket].append(entry)
        print(f"  [{bucket}] {artist[:25]} — {title[:35]} (ang={shabad_data['ang']}, "
              f"lines={len(shabad_data['shabad_lines'])})")

        total = sum(len(v) for v in buckets.values())
        if total >= TARGET_PER_BUCKET * len(buckets):
            break

    catalog = []
    for bucket, tracks in buckets.items():
        print(f"\n{bucket}: {len(tracks)} tracks")
        catalog.extend(tracks)

    print(f"\nTotal: {len(catalog)} tracks")
    return catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print catalog but don't write to file")
    args = parser.parse_args()

    catalog = build_catalog(dry_run=args.dry_run)

    if not catalog:
        print("ERROR: No tracks collected!")
        return

    if args.dry_run:
        print("\n[dry-run] Would write:")
        for t in catalog:
            print(f"  {t['style_bucket']:<8} {t['artist_name'][:25]:<25} {t['title'][:40]}")
        return

    out_path = Path(OUTPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(catalog)} tracks → {out_path}")

    # Summary by bucket
    buckets: dict[str, int] = {}
    for t in catalog:
        b = t["style_bucket"]
        buckets[b] = buckets.get(b, 0) + 1
    for b, n in sorted(buckets.items()):
        print(f"  {b}: {n}")


if __name__ == "__main__":
    main()
