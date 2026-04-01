# Kirtan Style Guide — Data Collection Phases

## Phased data collection strategy

Each kirtan style has different acoustic properties that affect Whisper alignment quality.
Rather than mixing all styles in one pass, we collect and pilot each style type separately
so we can tune alignment parameters per style before scaling.

## Style types (collection order)

### Phase A: Clean kirtan (collect first)
High-quality studio recordings with clear vocals and minimal background noise.
Best alignment quality — use these to validate the pipeline.

| Style | Description | Source keywords | Alignment difficulty |
|-------|-------------|-----------------|---------------------|
| **Studio** | Professional studio recordings | "studio", album tracks | Easy | Famous Ragi
| **Hazoori** | Darbar Sahib style, formal | "darbar sahib", Bhai Harjinder Singh, Prof Darshan Singh | Easy-Medium |
| **Puratan** | Classical raag-based, slow tempo | "raag", "classical", Bhai Surinder Singh Jodhpuri | Medium |

### Phase B: Mixed/live kirtan (collect second)
Live recordings with audience, hall reverb, varying mic quality.

| Style | Description | Source keywords | Alignment difficulty |
|-------|-------------|-----------------|---------------------|
| **Taksali** | Damdami Taksal style, precise santhya pronunciation | "taksal", "taksali" | Medium |
| **Live** | General gurdwara recordings | "gurdwara", "gurudwara", "diwan" | Medium-Hard |
| **Mixed** | Unclear or multi-style | Fallback bucket | Medium-Hard |

### Phase C: AKJ kirtan (collect last)
AKJ kirtan is significantly harder for the alignment pipeline due to its unique acoustic properties.
**Do not include AKJ data in early pilots — process after clean and live kirtan types are validated.**

| Style | Description | Source keywords | Alignment difficulty |
|-------|-------------|-----------------|---------------------|
| **AKJ** | Akhand Kirtani Jatha | "akj", "rainsbhai", "akhand-kirtan", Bhai Amolak Singh | Hard |

#### Why AKJ is harder
- **Fast tempo**: rapid word repetitions make sliding-window matching harder
- **Sangat joins in**: multiple voices overlap, confuses Whisper word boundaries
- **Repetitive loops**: same tuk sung 10-20 times back-to-back inflates window size
- **Call-and-response**: lead kirtani sings a line, sangat repeats — creates echo/overlap
- **Less melodic structure**: compared to raag-based kirtan, AKJ is rhythm-driven
  which gives Whisper fewer acoustic cues for word segmentation

#### AKJ-specific pipeline considerations (for when we get there)
- May need larger sliding window or repetition-aware deduplication
- Consider Whisper word confidence weighting — AKJ segments often have lower avg_confidence
- Possibly needs separate alignment thresholds (lower mapped_ratio acceptance)
- SikhNet playlist: `akhand-kirtan` (53 tracks, 96% have shabadId)

## Pilot findings (v2 test, 10 recordings)

From the first v2 word-level mapping test run:
- Clean kirtan: mapped_ratio near 1.0, alignment works well
- AKJ recordings: lower mapped_ratio, more unmapped words, needs tuning
- Total: 1,217 segments from 10 recordings (all styles mixed)

## Training phase mapping

| Data phase | Training phase | Style buckets |
|------------|---------------|---------------|
| Phase A (clean) | Phase 2 (clean) + Phase 3 (hazoori/puratan) | studio, hazoori, puratan |
| Phase B (live) | Phase 4 (live/noisy) | taksali, live, mixed |
| Phase C (AKJ) | Phase 4 (live/noisy) — added after Phase B validation | akj |
