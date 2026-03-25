# Surt — Data Collection (SikhNet + BaniDB)

## Why SikhNet + BaniDB
~60% of SikhNet tracks (~21,000 of ~35,000) have a `shabadId` field that maps
directly to BaniDB, which returns exact Unicode Gurmukhi text for every verse.

This gives us GROUND TRUTH labels — the training label is always the exact
BaniDB canonical text, not a Whisper approximation. No fuzzy matching needed.

Old broken approach: Whisper teacher → fuzzy corpus match → all scores 1.0 (garbage)
New approach: shabadId → BaniDB → exact shabad lines with ang, raag, writer

## Target dataset
- 700-900 hours raw kirtan from SikhNet with shabadId
- Quality-filter to ~400 hours
- Combined with ~90 hours sehaj path = ~500 hours total training data

## Implementation
Full scraper implemented at `src/data/sikhnet.py`.
Includes: checkpoint/resume, retry decorator, BaniDB cache, Gurmukhi ratio validation,
async downloads, parallel BaniDB fetching, HF Hub push in shards.

Run with:
```bash
python src/data/sikhnet.py pilot          # 100-track validation first
python src/data/sikhnet.py build-catalog  # full artist discovery + BaniDB enrichment
python src/data/sikhnet.py download       # async MP3 download
python src/data/sikhnet.py push-hf        # push to HF Hub in 500-track shards
python src/data/sikhnet.py stats          # print catalog summary
```

## APIs used

### SikhNet search (artist discovery)
```
GET https://play.sikhnet.com/api/search
  ?q=kirtan&type=audio&perPage=50&page=1
```
Response: `response[i].rowsStackedWidget.items[j].action.trackInfo`

### SikhNet artist tracks (bulk, preferred)
```
GET https://play.sikhnet.com/_next/data/{buildId}/artist/{slug}/tracks.json
```
BuildId changes on each SikhNet deploy — discover dynamically from page HTML.
Re-discover on 404 mid-crawl (handled automatically in scraper).

### SikhNet playlist
```
GET https://play.sikhnet.com/_next/data/{buildId}/playlist/{slug}.json
```
Known playlists: `akhand-kirtan` (53 AKJ tracks, 96% have shabadId)

### Audio download
```
https://www.sikhnet.com/gurbani/audio/play/{trackId}
  → 301 redirect → https://media.sikhnet.com/{Artist}%20-%20{Title}.mp3
```
Follow redirect to get direct MP3 URL.

### BaniDB shabad text
```
GET https://api.banidb.com/v2/shabads/{shabadId}
```
Key fields: `verses[].verse.unicode` (Gurmukhi text), `shabadInfo.source.pageNo` (ang),
`shabadInfo.raag.english`, `shabadInfo.writer.english`

## Filtering rules

### Include
- `shabadId` present and not null/0/-1
- Duration: 2 min (120s) to 90 min (5400s)
- BaniDB returns valid response (not 404)
- Gurmukhi ratio of BaniDB text ≥ 80%

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
|-------|-----------------|----------------|
| hazoori | Bhai Harjinder Singh, Prof Darshan Singh, "darbar sahib" | Phase 3 |
| puratan | Bhai Surinder Singh Jodhpuri, "classical", "raag" | Phase 3 |
| akj | Bhai Amolak Singh (AKJ), "akj", "rainsbhai", playlist/akhand-kirtan | Phase 4 |
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

## whisperX alignment (GPU step)
After downloading, audio is aligned on RunPod GPU pods (4 in parallel).
Each pod:
1. Runs whisperX forced alignment on its shard
2. Matches aligned segments to known `shabad_lines` from BaniDB
3. Produces tuk-level timestamps
4. Pushes aligned shard to HF Hub immediately (crash-safe)
5. Updates `alignment_status` to "aligned" or "rejected"

Training label is ALWAYS `shabad_lines` from BaniDB — never Whisper output.
Whisper is used only to get timestamps, not the text.

## Rate limiting
- SikhNet: 0.5s between requests, max 8 concurrent downloads
- BaniDB: 0.3s between requests, max 5 concurrent, full response cache
- User-Agent: identify as "GurbaniASR-DataPipeline/1.0"
- On 429: exponential backoff starting at 2s

## Storage
- Audio files: `data/audio/{recording_id}.mp3` on Hetzner volume
- Catalog: `data/manifests/sikhnet_catalog.json`
- BaniDB cache: `data/manifests/banidb_cache.json` (never re-fetch)
- Scraper checkpoint: `data/manifests/scraper_checkpoint.json` (resume state)
- Final dataset on HF Hub: `YOUR_HF_USERNAME/gurbani-asr-dataset`
