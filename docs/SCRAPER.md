# Surt — Data Collection (SikhNet + STTM Database)

## Why SikhNet + STTM database

~60% of SikhNet tracks (~21,000 of ~35,000) have a `shabadId` field that maps
directly to the local STTM `database.sqlite`, which contains exact Unicode
Gurmukhi text for every verse.

This gives us GROUND TRUTH labels — the training label is always the exact
canonical STTM text, not a Whisper approximation. No transcription needed.

**Critical design principle**: There is NO Whisper transcription in the data
pipeline. Since we already know the exact text (via shabadId → STTM database),
we only need WhisperX **forced alignment** to get tuk-level timestamps.

Old broken approach: Whisper teacher → fuzzy corpus match → all scores 1.0 (garbage)
New approach: shabadId → local database.sqlite → exact shabad lines with ang, raag, writer

## Target dataset

- 700-900 hours raw kirtan from SikhNet with shabadId
- Quality-filter to ~400 hours
- Combined with ~90 hours sehaj path = ~500 hours total training data

## Implementation

Full scraper implemented at `src/data/sikhnet.py`.
Includes: checkpoint/resume, retry decorator, STTM database lookups, Gurmukhi ratio
validation, async downloads, HF Hub push in shards.

Run with:

```bash
python src/data/sikhnet.py pilot          # 100-track validation first
python src/data/sikhnet.py build-catalog  # full artist discovery + STTM enrichment
python src/data/sikhnet.py download       # async MP3 download
python src/data/sikhnet.py push-hf        # push to HF Hub in 500-track shards
python src/data/sikhnet.py stats          # print catalog summary
```

## APIs used

### SikhNet search (artist discovery)

```text
GET https://play.sikhnet.com/api/search
  ?q=kirtan&type=audio&perPage=50&page=1
```

Response: `response[i].rowsStackedWidget.items[j].action.trackInfo`

### SikhNet artist tracks (bulk, preferred)

```text
GET https://play.sikhnet.com/_next/data/{buildId}/artist/{slug}/tracks.json
```

BuildId changes on each SikhNet deploy — discover dynamically from page HTML.
Re-discover on 404 mid-crawl (handled automatically in scraper).

### SikhNet playlist

```text
GET https://play.sikhnet.com/_next/data/{buildId}/playlist/{slug}.json
```

Known playlists: `akhand-kirtan` (53 AKJ tracks, 96% have shabadId)

### Audio download

```text
https://www.sikhnet.com/gurbani/audio/play/{trackId}
  → 301 redirect → https://media.sikhnet.com/{Artist}%20-%20{Title}.mp3
```

Follow redirect to get direct MP3 URL.

### STTM database lookup (local, no API)

Shabad text is looked up from the local `database.sqlite` file (151MB STTM database).
NO BaniDB API calls — everything is a local SQLite query via `lookup_shabad()`.

```python
# src/data/sikhnet.py — lookup_shabad(sttm_id)
# shabads.sttm_id (integer) maps to SikhNet's shabadId
# lines.gurmukhi uses STTM ASCII encoding → converted to Unicode via ascii_to_unicode()
```

Key fields from database: `lines.gurmukhi` (ASCII → Unicode), `lines.source_page` (ang),
`sections.name_english` (raag), `writers.name_english` (writer)

### Vishram markers in STTM text

The `lines.gurmukhi` column encodes vishram (pause) positions inline:

| Marker | Meaning | Coverage |
| --- | --- | --- |
| `;` | Primary vishram (yamki) — main pause in a pangti | 79,285 / 141,264 lines (56%) |
| `.` | Secondary vishram — smaller natural pause | 18,104 lines |

Example: `socY. soic n hoveI; jy socI lK vwr ]`

- Secondary pause after `socY`
- **Primary pause** after `hoveI` — this is where kirtanis split the line

The `vishraam_first_letters` column tracks these positions separately from `first_letters`.

**Used in alignment**: Lines with `;` are split into three match targets (full line, first half,
second half) because kirtanis often sing each half separately, pausing at the vishram.
All vishram markers are stripped from training labels.

## Filtering rules

### Include

- `shabadId` present and not null/0/-1
- Duration: 2 min (120s) to 90 min (5400s)
- Shabad exists in local database.sqlite
- Gurmukhi ratio of STTM text >= 80%

### Exclude — blacklist keywords (in title OR album)

```python
["akhand path", "akhand paath", "sehaj path", "sehaj paath", "sahaj path",
 "katha", "vichar", "vichaar", "teeka", "tika", "nitnem", "rehras sahib",
 "japji sahib", "sukhmani sahib", "asa di var", "lecture", "discourse",
 "dukh bhanjani", "ardas", "japji paath", "anand sahib paath"]
```

### Exclude — katha/discourse artists

```python
["Giani Sant Singh Maskeen", "Bhai Pinderpal Singh",
 "Sant Baba Gurbachan Singh", "Khalsa Prabandhak Jatha",
 "Bhai Sarbjit Singh Dhunda"]
```

### Deduplication

By `(shabadId, artistId)` pair — not just URL.
Same artist + same shabad in multiple albums = same audio, skip duplicates.

## Style classification

Each recording gets a `style_bucket` used to assign training phase:

| Style | Artists/keywords | Training phase |
| --- | --- | --- |
| hazoori | Bhai Harjinder Singh, Prof Darshan Singh, "darbar sahib" | Phase 3 |
| puratan | Bhai Surinder Singh Jodhpuri, "classical", "raag" | Phase 3 |
| akj | Bhai Amolak Singh (AKJ), "akj", "rainsbhai", akhand-kirtan | Phase 4 |
| taksali | Damdami Taksal, "taksal", "taksali" | Phase 4 |
| live | "gurdwara", "gurudwara", "diwan" | Phase 4 |
| mixed | No match | Phase 4 |

## Recording manifest schema

Each record in `data/manifests/sikhnet_catalog.json`:

```json
{
  "recording_id":    "md5(url)[:16]",
  "url":             "https://media.sikhnet.com/...",
  "title":           "...",
  "artist_name":     "...",
  "artist_id":       1234,
  "style_bucket":    "akj",
  "source_name":     "sikhnet",
  "sikhnet_track_id": 81282,
  "shabad_id":       2244,
  "shabad_lines":    ["ਸੂਹੀ ਮਹਲਾ ੪ ॥", "ਐ ਜੀ ਤੂ ਐਸਾ ..."],
  "ang":             757,
  "raag":            "Raag Soohee",
  "writer":          "Guru Ram Das Ji",
  "duration_sec":    1158.0,
  "local_path":      "data/audio/a1b2c3d4.mp3",
  "alignment_status": "pending",
  "phase":           4
}
```

## Forced alignment (GPU step — RunPod)

**There is NO Whisper transcription used as training labels.**
We already know the exact text for every recording (shabadId → database.sqlite).
Whisper large-v3 is used ONLY for timestamp extraction (`word_timestamps=True`).
The timestamped words are then matched to canonical tuks via matra-normalised F1 scoring.

### Why not WhisperX or pure forced alignment

- WhisperX has no Punjabi alignment model (not in supported language list)
- Standard forced alignment assumes text appears once in order — breaks for kirtan
  where tuks are repeated 2-5x and sung in halves at vishram pauses
- Whisper-as-timestamp-oracle + fuzzy matching handles repetitions and variable
  singing patterns naturally

### Alignment pipeline (`src/data/prepare_pilot.py`)

1. Loads canonical `shabad_lines` from the catalog (enriched from database.sqlite)
2. Splits each line at primary vishram (`;`) into [full, first_half, second_half] match targets
3. Runs Whisper large-v3 with `word_timestamps=True` on audio — timestamps only
4. Forward-scans Whisper words, matching each region to the best canonical tuk/half-tuk
   using matra-stripped F1 scoring (threshold ≥ 0.5)
5. Same tuk can match multiple times → captures kirtan repetitions as separate segments
6. Segments audio at matched boundaries, labels with canonical STTM text
7. Pushes aligned segments to HF Hub

Training label is ALWAYS canonical text from STTM database — never Whisper output.
Vishram markers (`;`, `.`) are stripped from all training labels.

## Rate limiting

- SikhNet: 0.5s between requests, max 8 concurrent downloads
- User-Agent: identify as "GurbaniASR-DataPipeline/1.0"
- On 429: exponential backoff starting at 2s

## Storage

- Audio files: `data/audio/{recording_id}.mp3` on Hetzner volume
- Catalog: `data/manifests/sikhnet_catalog.json`
- STTM database: `database.sqlite` (local, 151MB, checked into project)
- Scraper checkpoint: `data/manifests/scraper_checkpoint.json` (resume state)
- Final dataset on HF Hub: `surindersinghssj/gurbani-asr-dataset`
