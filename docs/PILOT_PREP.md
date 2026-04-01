# Surt — Pilot Data Preparation

## Goal

Take 93 pilot recordings (23.1 hours raw audio) with known `shabadId` and produce
training-ready `(audio_segment, canonical_text)` pairs where:

- Audio is segmented at tuk boundaries (1-30 seconds each)
- Labels are ALWAYS canonical STTM text — never Whisper output
- Vishram-aware: each pangati yields multiple match targets

## Why 3 pipelines

We run all 3 to compare results on the same 93-track pilot set:

1. **Pipeline 1 (Whisper large-v2 GPU)** — the gold standard. Plain Whisper large-v2
   handles noisy kirtan audio well and produces accurate enough transcription that
   fuzzy-matching to vishram-partitioned STTM tuks is highly reliable. This is our
   baseline for what "good" alignment looks like.

2. **Pipeline 2 (faster-whisper CPU)** — the validation test. Can CTranslate2's
   faster-whisper with INT8 quantisation match plain Whisper large-v2's quality? If
   yes, we use it for full-scale (700+ hours) because it's cheaper and needs no GPU.
   If not, we know where the gap is.

3. **Pipeline 3 (Demucs + WhisperX)** — the experiment. Does vocal separation via
   Demucs improve alignment on instrument-heavy tracks? This tests whether removing
   harmonium/tabla before alignment helps or hurts.

## The one rule

**Whisper is a timestamp oracle, not a transcriber.**
We already know every word (shabadId → database.sqlite). Whisper tells us *when*
those words occur in the audio. The training label is always the canonical STTM text.

---

## STTM Tuk Partitioning (all pipelines)

Every canonical line from `database.sqlite` is expanded into match targets using
vishram markers before alignment. This is shared across all 3 pipelines.

### Vishram markers in STTM

| Marker | Meaning | Example |
| --- | --- | --- |
| `;` | Primary vishram (yamki) — main pause | `socY soic n hoveI; jy socI lK vwr ]` |
| `.` | Secondary vishram — smaller pause | `socY. soic n hoveI; jy socI lK vwr ]` |

### Partition rules

For each pangati, generate these match targets:

1. **Full pangati** — the complete line with all markers stripped
2. **Split at each primary vishram (`;`)** — N vishrams produce N+1 segments
3. **If secondary vishrams (`.`) exist within a partition** — further split those too

**Example**: `AwpY. Awip aupdysI; Awip buJwvxhwrw; kir ikrpw ]`

This has 2 primary (`;`) and 1 secondary (`.`) vishram:

| Target | Text |
| --- | --- |
| Full | `AwpY Awip aupdysI Awip buJwvxhwrw kir ikrpw` |
| Part 1 (before 1st `;`) | `AwpY Awip aupdysI` |
| Part 2 (between `;`) | `Awip buJwvxhwrw` |
| Part 3 (after last `;`) | `kir ikrpw` |
| Sub-part 1a (before `.`) | `AwpY` |
| Sub-part 1b (after `.`) | `Awip aupdysI` |

Each partition must have >= 2 words to be a valid match target. Single-word
fragments are discarded. All vishram markers are stripped from the final label.

**Why this matters**: Kirtanis sing pangatis in chunks — sometimes a full line,
sometimes half at a vishram, sometimes just a phrase. More partition targets = more
segments captured = more training data from the same audio.

---

## Step-by-step (shared across all pipelines)

```
Step 1: Load audio + canonical tuks from catalog
Step 2: Get word-level timestamps from audio (method varies by pipeline)
Step 3: Fuzzy-match timestamped words to STTM tuks → replace with ground truth (Option B)
Step 4: Segment audio at matched boundaries → FLAC + canonical label
Step 5: Quality filter → push to HF Hub
```

### Step 3 — Option B: Fuzzy-match → ground truth replacement

This is the approach for all 3 pipelines. The timestamped words from Whisper are
**not used as labels**. They serve only as an anchor to locate where each canonical
tuk occurs in the audio:

1. Matra-strip both Whisper words and STTM tuk words (consonant skeleton matching)
2. Slide a window over Whisper words, compute F1 overlap with each tuk partition
3. Best-scoring tuk above threshold (F1 >= 0.5) claims that audio region
4. The **label** for that segment is the canonical STTM text, not the Whisper text
5. Same tuk can match multiple times (handles kirtan repetitions)

```
Whisper output (timestamps only):     "ਸੋਚੈ ਸੋਚ ਨ ਹੋਵਈ ਜੇ ..."  @ 12.3s-18.7s
STTM canonical (ground truth):        "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ ॥"
Training label written:                "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ"  ← always STTM
```

---

## Pipeline 1 — Whisper large-v2 on GPU

**Engine**: `openai/whisper-large-v2` (plain HuggingFace/OpenAI, NOT faster-whisper)
**Hardware**: 3-4 RunPod GPUs (16 GB or 24 GB VRAM each)
**Parallelism**: Split recordings across GPUs, each processes independently

### Why plain Whisper large-v2 is the gold standard

- **Noise-robust**: Whisper large-v2 handles kirtan noise (harmonium, tabla, sangat,
  hall reverb) better than any other open model. It produces accurate enough
  transcription even on noisy recordings that the fuzzy-match to STTM tuks with
  vishram-partitioned text is highly reliable.
- **Accurate content = easy matching**: Because the Whisper output closely matches
  the actual Gurbani words being sung, the F1 overlap with STTM tuk partitions
  (full pangati, vishram halves, sub-parts) is high. This means more tuks get
  matched, more repetitions are captured, and fewer segments are lost to low scores.
- **No quantisation artifacts**: CTranslate2 (faster-whisper) can lose word-level
  timestamp precision due to INT8 quantisation. Plain Whisper preserves full
  precision in both transcription content and timestamp placement.
- **Battle-tested on Indic languages**: large-v2 has the most community validation
  on Punjabi and Hindi audio. large-v3 is newer but less proven on our domain.

### Hardware requirements

| Setting | Value |
| --- | --- |
| Model | `openai/whisper-large-v2` (1.55B params) |
| Precision | float16 |
| VRAM per GPU | ~10 GB loaded, ~14 GB peak during inference |
| Min GPU | 16 GB VRAM |
| Recommended | 24 GB VRAM for headroom |
| Workers | 3-4 GPUs running in parallel |

### Flow

```
                    ┌─── GPU 1: recordings 1-25 ───┐
                    │                               │
93 recordings ──────┼─── GPU 2: recordings 26-50 ──┼──→ merge manifests
(23.1 hours)        │                               │       │
                    ├─── GPU 3: recordings 51-75 ──┤       ▼
                    │                               │   fuzzy-match
                    └─── GPU 4: recordings 76-93 ──┘   to STTM tuks
                                                          │
                              Whisper large-v2             ▼
                              word_timestamps=True    segment audio
                              language="pa"           + canonical labels
                              beam_size=5                  │
                              fp16=True                    ▼
                                                      push to HF Hub
```

### Per-GPU processing

```python
import whisper

model = whisper.load_model("large-v2", device="cuda")

result = model.transcribe(
    audio_path,
    language="pa",
    word_timestamps=True,
    beam_size=5,
    temperature=0,
    condition_on_previous_text=True,
    fp16=True,
)
# result["segments"][i]["words"] → list of {word, start, end, probability}
```

### Timing estimate

- Whisper large-v2 on 16/24 GB GPU: ~0.3x realtime (1 hour audio ≈ 18 min)
- 23.1 hours / 4 GPUs = ~5.8 hours each → ~1.7 hours wall time
- With overhead (loading, saving, network): ~2-2.5 hours total

### Pros / Cons

| Pros | Cons |
| --- | --- |
| Most accurate word timestamps | Requires GPU (not free) |
| Battle-tested on Indic languages | Slower than faster-whisper |
| No quantisation artifacts | Needs RunPod or cloud GPU |
| Best for establishing ground-truth timestamps | 10 GB VRAM minimum |

---

## Pipeline 2 — faster-whisper large-v2/v3 on CPU

**Engine**: `faster-whisper` (CTranslate2 backend) with `large-v2` or `large-v3`
**Hardware**: 3-4 CPU machines (12-16 GB RAM each)
**Parallelism**: Split recordings across CPUs, each processes independently

### Why test faster-whisper

The question we're answering: **Can faster-whisper match plain Whisper large-v2's
transcription quality on noisy kirtan audio?**

- If yes → use faster-whisper for full-scale (700+ hours). No GPU needed, much cheaper.
- If no → we know the quality gap and can decide: accept it, or use Pipeline 1 for
  the full dataset (more expensive but better alignment).

Specific things to compare against Pipeline 1:

- Transcription accuracy on noisy tracks (word-level match to Pipeline 1 output)
- F1 match scores against STTM tuks (are they close enough?)
- Number of matched segments per recording (are we losing coverage?)
- Timestamp precision (do segments start/end at the right place?)

### Hardware requirements

| Setting | Value |
| --- | --- |
| Model | `large-v2` or `large-v3` via faster-whisper |
| Compute type | `int8` (CPU) or `float32` (slower, more accurate) |
| RAM per worker | ~6 GB loaded (int8), ~12 GB peak |
| Min RAM | 12 GB per machine |
| Recommended | 16 GB for headroom |
| Workers | 3-4 CPU processes |
| CPU | 4+ cores recommended per worker |

### Flow

```
                    ┌─── CPU 1: recordings 1-25 ───┐
                    │                               │
93 recordings ──────┼─── CPU 2: recordings 26-50 ──┼──→ merge manifests
(23.1 hours)        │                               │       │
                    ├─── CPU 3: recordings 51-75 ──┤       ▼
                    │                               │   fuzzy-match
                    └─── CPU 4: recordings 76-93 ──┘   to STTM tuks
                                                          │
                          faster-whisper large-v2          ▼
                          word_timestamps=True        segment audio
                          compute_type="int8"         + canonical labels
                          language="pa"                    │
                          beam_size=3                      ▼
                                                      push to HF Hub
```

### Per-worker processing

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v2", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    audio_path,
    language="pa",
    word_timestamps=True,
    beam_size=3,             # lower beam for CPU speed
    vad_filter=True,         # skip silence (saves CPU time)
    condition_on_previous_text=True,
)

all_words = []
for seg in segments:
    if seg.words:
        for w in seg.words:
            all_words.append({"word": w.word, "start": w.start, "end": w.end,
                              "probability": w.probability})
```

### Timing estimate

- faster-whisper large-v2 INT8 on CPU (4-core): ~1.5-2x realtime
- 23.1 hours / 4 workers = ~5.8 hours each → ~9-12 hours wall time per worker
- With 4 workers in parallel: ~9-12 hours wall time
- Hetzner VPS already available, or 3-4 cheap CPU machines

### large-v2 vs large-v3

| | large-v2 | large-v3 |
| --- | --- | --- |
| Indic language quality | Excellent, battle-tested | Improved but less tested |
| Word timestamp reliability | Very stable | Occasionally misaligned on long audio |
| CTranslate2 support | Full | Full |
| Recommendation | **Use for pilot** | Try if v2 timestamps are poor |

### Pros / Cons

| Pros | Cons |
| --- | --- |
| No GPU needed | Slower than GPU (1.5-2x realtime) |
| Runs on existing Hetzner VPS | INT8 may reduce timestamp precision slightly |
| Very cheap / free | Higher beam sizes impractical on CPU |
| Easy to scale (just add machines) | 12+ GB RAM required per worker |

---

## Pipeline 3 — Demucs + Silence + WhisperX Force-Align

**Engine**: Demucs (vocal separation) → silence-based segmentation → WhisperX forced alignment
**Hardware**: 1 GPU for Demucs (4-8 GB VRAM), CPU for the rest
**Key difference**: This pipeline separates vocals BEFORE alignment

### Why experiment with Demucs

The question we're answering: **Does removing instruments before alignment improve
segment quality on kirtan audio?**

- Demucs (htdemucs) separates vocals from harmonium/tabla/drums
- On clean vocals, silence gaps naturally mark phrase boundaries (vishram pauses)
- WhisperX forced alignment on clean vocals might give more precise timestamps
- But: Demucs can distort vocals, add artifacts, or remove vocal harmonics that
  Whisper uses for recognition. We need to test whether it helps or hurts.

Compare against Pipeline 1:

- Do we get more or fewer matched segments?
- Are match scores higher on instrument-heavy recordings specifically?
- Does silence-based segmentation find vishram boundaries more naturally?
- Any vocal distortion artifacts that break the fuzzy-match?

### Important: WhisperX with STTM tuks (Option B)

WhisperX has no Punjabi alignment model in its supported languages. However, we
use it in **forced alignment mode** where we provide the text to align — it does
not need to transcribe. The text comes from STTM tuks (canonical ground truth).

WhisperX forced alignment with provided text:
1. Takes clean vocal audio (post-Demucs)
2. Takes the STTM canonical text as input
3. Aligns the known text to the audio signal
4. Returns word-level timestamps for the STTM words

If WhisperX Punjabi alignment fails, fall back to Pipeline 1/2's fuzzy-match
approach on the Demucs-separated vocals.

### Hardware requirements

| Component | Hardware | Notes |
| --- | --- | --- |
| Demucs (htdemucs) | 1 GPU, 4-8 GB VRAM | Or CPU (~10x slower) |
| Silence detection | CPU only | pydub/librosa, trivial |
| WhisperX align | CPU or GPU | Forced alignment is lightweight |

### Flow

```
                         Demucs htdemucs
93 recordings ─────────→ vocal separation ─────→ vocals-only audio
(23.1 hours)             (GPU, ~0.1x RT)              │
                                                       ▼
                                              silence detection
                                              (energy threshold +
                                               min_silence=500ms)
                                                       │
                                                       ▼
                                              phrase-level chunks
                                              (natural breath boundaries)
                                                       │
                                                       ▼
                                       ┌───────────────┴────────────────┐
                                       │                                │
                                  WhisperX                        Fallback:
                                  force-align                     Whisper timestamps
                                  with STTM tuks                  + fuzzy-match
                                  (Option B)                      (if alignment fails)
                                       │                                │
                                       └───────────────┬────────────────┘
                                                       ▼
                                              segment audio at
                                              tuk boundaries
                                              + canonical labels
                                                       │
                                                       ▼
                                                  push to HF Hub
```

### Step-by-step

**Step 1: Demucs vocal separation**

```python
from demucs.pretrained import get_model
from demucs.apply import apply_model
import torch, torchaudio

model = get_model("htdemucs")
model.to("cuda")

wav, sr = torchaudio.load(audio_path)
wav = wav.unsqueeze(0)  # add batch dim

with torch.no_grad():
    sources = apply_model(model, wav, device="cuda")
    # sources shape: (batch, 4_sources, channels, samples)
    # Index 3 = vocals (htdemucs order: drums, bass, other, vocals)
    vocals = sources[0, 3]

torchaudio.save("vocals.wav", vocals.cpu(), sr)
```

**Step 2: Silence-based segmentation**

```python
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

audio = AudioSegment.from_wav("vocals.wav")
nonsilent_ranges = detect_nonsilent(
    audio,
    min_silence_len=500,    # 500ms silence = phrase break
    silence_thresh=-40,      # dBFS threshold (tune per recording)
    seek_step=10,
)
# nonsilent_ranges = [(start_ms, end_ms), ...]
# Each range is a natural phrase — corresponds roughly to a tuk or half-tuk
```

**Step 3: WhisperX forced alignment with STTM tuks (Option B)**

```python
import whisperx

# Load alignment model
align_model, align_metadata = whisperx.load_align_model(
    language_code="pa",
    device="cuda",
)

# Provide STTM canonical text as the transcript to align
# (NOT asking WhisperX to transcribe — giving it the known text)
sttm_text = " ".join(shabad_lines)  # canonical tuks joined

transcript_segments = [{
    "text": sttm_text,
    "start": 0.0,
    "end": audio_duration,
}]

result = whisperx.align(
    transcript_segments,
    align_model,
    align_metadata,
    vocals_audio,
    device="cuda",
)
# result["word_segments"] → word-level timestamps for STTM words

# If Punjabi alignment model unavailable/poor quality:
# Fall back to Pipeline 1/2 fuzzy-match on the Demucs vocals
```

**Step 4: Map aligned words to tuk partitions**

Same fuzzy-match logic as Pipelines 1 & 2, but operating on cleaner (Demucs)
audio with potentially better word boundaries from silence detection.

### Timing estimate

- Demucs htdemucs on GPU: ~0.1x realtime (1 hour audio ≈ 6 min)
- 23.1 hours → ~2.3 hours for Demucs
- Silence detection: negligible (seconds)
- WhisperX alignment: ~0.2x realtime → ~4.6 hours
- **Total**: ~3-4 hours
- **Total**: ~3-4 hours on 1 GPU + CPU

### Pros / Cons

| Pros | Cons |
| --- | --- |
| Cleanest audio signal (no instruments) | Extra Demucs step (GPU or slow CPU) |
| Best for music-heavy recordings | More moving parts / complexity |
| Silence detection gives natural phrase boundaries | WhisperX Punjabi support uncertain |
| Can rescue tracks that fail in Pipeline 1/2 | Demucs can occasionally distort vocals |

---

## Comparison: All 3 Pipelines

| | Pipeline 1 | Pipeline 2 | Pipeline 3 |
| --- | --- | --- | --- |
| **Engine** | Whisper large-v2 (plain) | faster-whisper large-v2/v3 | Demucs + WhisperX |
| **Hardware** | 3-4 GPUs (16-24 GB) | 3-4 CPUs (12-16 GB) | 1 GPU + CPU |
| **Timestamp source** | Whisper word_timestamps | faster-whisper word_timestamps | WhisperX forced align |
| **Matching** | Fuzzy-match → STTM (Option B) | Fuzzy-match → STTM (Option B) | Force-align STTM text (Option B) |
| **Labels** | Canonical STTM text | Canonical STTM text | Canonical STTM text |
| **Wall time** | ~2-2.5 hours | ~9-12 hours | ~3-4 hours |
| **Timestamp quality** | Highest | Good (INT8 may lose some) | Best on clean vocals |
| **Best for** | Gold-standard pilot | Budget / no-GPU runs | Instrument-heavy tracks |
| **Complexity** | Low | Low | Medium |

---

## Pilot strategy

Run all 3 pipelines on the same 93 tracks. Compare results to decide what to use
for full-scale (700+ hours):

1. **Pipeline 1 first** — establishes the gold-standard baseline. This is what
   "good" alignment looks like. All other pipelines are measured against this.

2. **Pipeline 2 second** — compare output quality against Pipeline 1. If match
   scores and segment counts are within ~5%, use faster-whisper for full-scale
   (saves GPU cost). If there's a meaningful gap, Pipeline 1 is worth the cost.

3. **Pipeline 3 third** — compare specifically on instrument-heavy tracks (hazoori,
   puratan styles). If Demucs improves alignment on those tracks vs Pipeline 1,
   use it as a rescue path for difficult recordings in full-scale.

### Decision criteria after pilot

| Result | Decision for full-scale |
| --- | --- |
| Pipeline 2 ≈ Pipeline 1 | Use Pipeline 2 (cheapest, no GPU) |
| Pipeline 2 < Pipeline 1 by >5% | Use Pipeline 1 (GPU cost justified) |
| Pipeline 3 rescues instrument-heavy tracks | Use Pipeline 1/2 primary + Pipeline 3 rescue |
| Pipeline 3 no better than Pipeline 1 | Drop Demucs entirely |

---

## Output schema (all pipelines)

Each segment produces one row:

```json
{
  "audio_path": "data/segments/{recording_id}_{seq:04d}.flac",
  "segment_id": "{recording_id}_{seq:04d}",
  "recording_id": "a1b2c3d4",
  "tuk_index": 3,
  "start": 12.345,
  "end": 18.720,
  "duration": 6.375,
  "training_label": "ਸੋਚੈ ਸੋਚਿ ਨ ਹੋਵਈ ਜੇ ਸੋਚੀ ਲਖ ਵਾਰ",
  "label_source": "sttm_database",
  "match_score": 0.82,
  "avg_confidence": 0.71,
  "repetition": 0,
  "partition_type": "full | first_half | second_half | vishram_part_N",
  "pipeline": "whisper_gpu | faster_whisper_cpu | demucs_whisperx",
  "ang": 1,
  "style_bucket": "hazoori",
  "shabad_id": 2244
}
```
