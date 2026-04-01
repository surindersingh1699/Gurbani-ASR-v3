# Whisper Model Comparison: 7 Configs × 4 Styles on Parallel RunPod GPUs

## Goal

Compare 3 Whisper models and 4 decoding-parameter variations on 4 kirtan styles
(hazoori, studio, AKJ, puratan). All with Mool Mantar prompt, VAD off.
3 pods run in parallel (1 model per GPU, turbo pod runs 5 configs sequentially).
Results pushed to HF Hub as playable dataset with 7 splits.

---

## The 7 Configurations

### Models (baseline configs)

| # | Config | Model | Key question |
|---|--------|-------|-------------|
| 1 | `large_v2` | large-v2 | Legacy baseline — how does the oldest large model compare? |
| 2 | `large_v3` | large-v3 | Does v3 improve Gurmukhi timestamps over v2? |
| 3 | `v3_turbo_base` | large-v3-turbo | Distilled model — does speed come at quality cost? |

### Turbo decoding variations (all use large-v3-turbo)

| # | Config | Change from baseline | Key question |
|---|--------|---------------------|-------------|
| 4 | `v3_turbo_rep` | + `repetition_penalty=1.1` | Does penalising repetition reduce Gurbani hallucination loops? |
| 5 | `v3_turbo_temp0` | + `temperature=0` (greedy) | Does greedy decoding improve timestamp precision? |
| 6 | `v3_turbo_beam5` | + `beam_size=5` (baseline=3) | Do more beams improve alignment quality enough to justify cost? |
| 7 | `v3_turbo_nocond` | + `condition_on_previous_text=False` | Does disabling context carryover reduce error propagation? |

### Transcription settings per config

```python
# Shared across all 7 configs
COMMON = dict(language="pa", word_timestamps=True, vad_filter=False, initial_prompt=MOOL_MANTAR)

CONFIGS = {
    "large_v2":        {**COMMON, "beam_size": 5, "condition_on_previous_text": True},
    "large_v3":        {**COMMON, "beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_base":   {**COMMON, "beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_rep":    {**COMMON, "beam_size": 5, "condition_on_previous_text": True,  "repetition_penalty": 1.1},
    "v3_turbo_temp0":  {**COMMON, "beam_size": 5, "condition_on_previous_text": True,  "temperature": 0},
    "v3_turbo_beam5":  {**COMMON, "beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_nocond": {**COMMON, "beam_size": 5, "condition_on_previous_text": False},
}
```

---

## Architecture

```
Local (launcher)
  │
  ├─ 0. Create templates if missing (one-time, stores template IDs)
  ├─ 1. Search SikhNet for 4 tracks (1 per style) → test_catalog.json
  │
  │  ┌─── PARALLEL ──────────────────────────────────────────────────────┐
  │  │                                                                    │
  ├─ 2a. Create 3 pods from templates    2b. Create tarball locally       │
  │       (templates auto-run startup:        (code + database.sqlite     │
  │        apt, pip, model download)           + test_catalog.json)       │
  │  │                                                                    │
  │  └────────────────────────────────────────────────────────────────────┘
  │       ↓ pods ready (model warm in cache)    ↓ tarball ready
  │
  ├─ 3. SCP tarball to all 3 pods in parallel
  ├─ 4. SSH run on each pod:
  │     Pod A: python3 scripts/whisper_model_comparison.py --model large-v2 --configs large_v2
  │     Pod B: python3 scripts/whisper_model_comparison.py --model large-v3 --configs large_v3
  │     Pod C: python3 scripts/whisper_model_comparison.py --model large-v3-turbo \
  │              --configs v3_turbo_base,v3_turbo_rep,v3_turbo_temp0,v3_turbo_beam5,v3_turbo_nocond
  ├─ 5. Monitor all 3 — collect reports incrementally as configs complete
  └─ 6. Print summary, terminate all 3 pods

Estimated wall clock: ~12-15 min (Pod C is bottleneck — 5 sequential runs on same model)
Setup wait eliminated: template startup (apt + pip + model download) runs during tarball creation.
```

Pod C loads `large-v3-turbo` once and runs all 5 turbo configs sequentially on the
same 4 tracks. Model loading is ~30s (from cache, no download); each 4-track run
is ~2 min on turbo. Total Pod C time: ~12 min (load + 5×2 min runs + overhead).

---

## Files to Create

### 1. `scripts/whisper_model_comparison.py` — Pod script (runs on RunPod)

Self-contained script. Takes `--model` (which HF model to load) and `--configs`
(comma-separated list of config names to run with that model).

**Flow:**

1. Read `data/manifests/test_catalog.json` (4 tracks)
2. Download audio from SikhNet (4 files, ~30s)
3. Load assigned faster-whisper model once (from HF cache — pre-downloaded by template startup)
4. For each config in `--configs`:
   - For each track: transcribe with config params → align → cut FLAC segments
   - Push to HF Hub as named split (e.g., `v3_turbo_rep`)
5. Write `logs/COMPLETE` marker

**Key functions:**

- `download_tracks(catalog)` — download 4 audio files
- `transcribe_track(model, audio, **config_params)` → `(word_dicts, raw_whisper_text)`
- `get_whisper_text_for_segment(word_dicts, start, end)` → string — post-hoc whisper text per segment
- `align_and_segment(word_dicts, whisper_text, audio, shabad_lines, track_meta, config_name)` → segment records
- `push_to_hf(segments, config_name)` — push as named split

### 2. `scripts/launch_comparison.py` — Launcher (runs locally)

Orchestrates 3 parallel pods. Steps:

1. Search SikhNet for 4 test tracks → write `data/manifests/test_catalog.json`
2. Ensure RunPod templates exist (create if missing)
3. Create tarball with `database.sqlite` included
4. Launch 3 pods from templates (parallel `runpod.create_pod()`)
5. Wait for all 3 pods ready
6. SCP tarball + SSH run in parallel (using threading)
7. Poll `logs/COMPLETE` on each pod
8. Print comparison summary
9. Terminate all pods (unless `--keep-pods`)

---

## RunPod Templates (pre-created to eliminate setup wait)

**Problem**: Without templates, each pod spends ~10-15 min on apt install, pip install,
and model download before any transcription begins.

**Solution**: Create 3 custom templates once. The template startup script runs automatically
when a pod is provisioned — by the time we SCP the tarball, the model is already in
HF cache and all deps are installed. Pod goes from "provisioned" to "ready to transcribe"
in ~30s instead of ~12 min.

### Template creation (run once via launcher `--create-templates`)

```python
import requests, os

API = "https://api.runpod.io/graphql"
HEADERS = {"Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"}
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"

TEMPLATES = [
    {
        "name": "surt-cmp-v2",
        "model_repo": "Systran/faster-whisper-large-v2",
    },
    {
        "name": "surt-cmp-v3",
        "model_repo": "Systran/faster-whisper-large-v3",
    },
    {
        "name": "surt-cmp-turbo",
        "model_repo": "Systran/faster-whisper-large-v3-turbo",
    },
]

for t in TEMPLATES:
    startup = (
        "apt-get update -qq && apt-get install -y -qq ffmpeg > /dev/null && "
        "pip install -q faster-whisper>=1.1.0 scipy numpy soundfile requests "
        "datasets huggingface-hub tqdm python-dotenv && "
        f"python3 -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{t['model_repo']}')\""
    )
    resp = requests.post(API, headers=HEADERS, json={"query": """
        mutation {{
            saveTemplate(input: {{
                name: "{name}"
                imageName: "{image}"
                containerDiskInGb: 30
                startScript: "{startup}"
                isPublic: false
            }}) {{ id name }}
        }}
    """.format(name=t["name"], image=IMAGE, startup=startup)})
    print(f"Created template: {t['name']} → {resp.json()}")
```

### Template details

| Template | Name | Model Pre-downloaded | Docker Image |
|----------|------|---------------------|-------------|
| A | `surt-cmp-v2` | `Systran/faster-whisper-large-v2` (~3GB) | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` |
| B | `surt-cmp-v3` | `Systran/faster-whisper-large-v3` (~3GB) | same |
| C | `surt-cmp-turbo` | `Systran/faster-whisper-large-v3-turbo` (~1.5GB) | same |

### What the startup script does (runs before our code)

1. `apt-get install ffmpeg` — needed for audio decoding (~20s)
2. `pip install faster-whisper datasets ...` — all Python deps (~40s)
3. `snapshot_download('Systran/faster-whisper-...')` — downloads CTranslate2 model to HF cache (~2-5 min depending on model size)

Total startup: ~3-6 min (runs while we create tarball + SCP). By the time our SSH
command fires, model is warm in cache. First `WhisperModel(...)` call takes ~30s
(loads from disk, no download).

### Pod launch (uses template IDs)

```python
# Launcher stores template IDs after creation
TEMPLATE_IDS = {
    "large-v2": "abc123",      # surt-cmp-v2
    "large-v3": "def456",      # surt-cmp-v3
    "large-v3-turbo": "ghi789" # surt-cmp-turbo
}

for model, template_id in TEMPLATE_IDS.items():
    runpod.create_pod(
        name=f"surt-cmp-{model}",
        gpu_type_id="NVIDIA RTX 4000 Ada Generation",  # 20GB, cheapest Ada
        template_id=template_id,   # ← model + deps pre-cached
        cloud_type="COMMUNITY",
        container_disk_in_gb=30,
    )
```

**GPU**: RTX 4000 Ada (20GB) or L4 (24GB). All 3 models fit in <10GB float16.

**Container disk**: 30GB (model ~3GB + code + audio + segments).

---

## Track Selection Strategy (in launcher)

Uses SikhNet API from `src/data/sikhnet.py`. Searches for known artists per style:

| Style | Artist Slugs to Try | Duration Target |
|-------|--------------------|--------------:|
| hazoori | Bhai Harjinder Singh Srinagar (use existing catalog entry) | 5-12 min |
| studio | `bhai-harcharan-singh-khalsa`, `bhai-manpreet-singh-kanpuri` | 5-12 min |
| akj | `bhai-niranjan-singh`, `akhand-kirtani-jatha` | 5-12 min |
| puratan | `prof-kartar-singh`, `bhai-balbir-singh` | 5-12 min |

**Filter**: must have `shabadId`, duration 3-15 min, has canonical lines from STTM.

**Fallback**: If a style can't be found, use SikhNet search API with style-specific queries.

---

## HF Dataset Schema

**Repo**: `surindersinghssj/gurbani-asr-model-comparison`

**Splits** (7): `large_v2`, `large_v3`, `v3_turbo_base`, `v3_turbo_rep`, `v3_turbo_temp0`, `v3_turbo_beam5`, `v3_turbo_nocond`

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `whisper_text` | string | Raw Whisper output for this segment's time range |
| 2 | `canonical_shabad` | string | Full shabad (all tuks joined) |
| 3 | `canonical_line` | string | The specific tuk matched by alignment |
| 4 | `training_text` | string | Portion of tuk actually sung (first→last anchor span) |
| 5 | `audio` | Audio(16kHz) | Playable FLAC segment |
| 6 | `config_name` | string | Config identifier (e.g. `v3_turbo_rep`) |
| 7 | `model_name` | string | `large-v2` / `large-v3` / `large-v3-turbo` |
| 8 | `segment_id` | string | `{recording_id}_{seq:04d}` |
| 9 | `recording_id` | string | Hash of track URL |
| 10 | `tuk_index` | int32 | Which tuk in the shabad |
| 11 | `start` | float32 | Segment start (seconds) |
| 12 | `end` | float32 | Segment end (seconds) |
| 13 | `duration` | float32 | Segment duration |
| 14 | `match_score` | float32 | Alignment coverage score |
| 15 | `avg_confidence` | float32 | Mean Whisper word probability |
| 16 | `match_method` | string | `first_letter` / `consonant_sim` |
| 17 | `repetition` | int32 | Tuk repetition count |
| 18 | `ang` | int32 | Page in SGGS |
| 19 | `raag` | string | From STTM database |
| 20 | `writer` | string | From STTM database |
| 21 | `style_bucket` | string | `hazoori` / `studio` / `akj` / `puratan` |
| 22 | `artist_name` | string | From SikhNet |
| 23 | `source_url` | string | SikhNet audio URL |

---

## Whisper Text Capture Strategy

`alignment.py`'s `match_words_to_tuks()` returns `start_sec`/`end_sec` per segment.
Post-hoc scan of `word_dicts` to get raw Whisper words in that time range:

```python
def get_whisper_text_for_segment(word_dicts, start_sec, end_sec):
    return " ".join(
        w["word"] for w in word_dicts
        if w["start"] >= start_sec - 0.05 and w["end"] <= end_sec + 0.05
    )
```

---

## What We're Comparing

### Per-config metrics
- Total segments produced (more = better coverage)
- Mean/median `match_score` (how well Whisper words match STTM tuks)
- Mean/median `avg_confidence` (Whisper's own confidence)
- Tuk coverage: % of canonical tuks matched at least once
- Repetitions captured (kirtan repeats same line — more = more training data)

### Comparison groups

#### Group A: Model comparison (configs 1-3, identical settings)

| Comparison | Tests | Answers |
|-----------|-------|---------|
| v2 vs v3 | Does v3's larger training data improve Gurmukhi? | Timestamp precision, match_score delta |
| v2 vs turbo | Can distilled turbo match full v2? | Quality vs speed tradeoff |
| v3 vs turbo | Does distillation lose v3's improvements? | Whether turbo inherits v3's Gurmukhi quality |

#### Group B: Decoding parameter impact (configs 3-7, same turbo model)

| Comparison | Tests | If it helps | If it hurts |
|-----------|-------|------------|-------------|
| base vs rep_penalty | Does `repetition_penalty=1.1` reduce hallucination loops? | Use for full-scale — Gurbani has real repetition so penalty must be gentle | Penalty disrupts kirtan repetition detection |
| base vs temp0 | Does greedy (`temperature=0`) improve precision? | Use greedy — fewer random word insertions | Greedy misses valid alternatives |
| base vs beam5 | Do more beams (5 vs 3) improve alignment? | Worth the ~40% speed cost | Stick with beam_size=3 |
| base vs nocond | Does disabling `condition_on_previous_text` reduce error propagation? | Use for long tracks where errors compound | Context helps maintain Gurmukhi script consistency |

### Decision matrix

| Result | Decision for full-scale (700+ hrs) |
|--------|-----------------------------------|
| Turbo ≈ v2/v3 (within ~5% match_score) | Use turbo (fastest, cheapest) |
| v3 > v2 > turbo | Use v3 (quality matters for training labels) |
| v2 ≈ v3 > turbo | Use v2 (proven, no turbo quality loss matters) |
| Any turbo variation > turbo base by >3% | Adopt that setting for full-scale |
| Multiple turbo settings help | Stack compatible improvements (e.g. rep_penalty + temp0) |

---

## Cost Breakdown

| Pod | GPU | Configs | $/hr (community) | Run time | Cost |
|-----|-----|---------|-------------------|----------|------|
| A: large-v2 | RTX 4000 Ada 20GB | 1 | ~$0.16 | ~8 min | ~$0.02 |
| B: large-v3 | RTX 4000 Ada 20GB | 1 | ~$0.16 | ~8 min | ~$0.02 |
| C: large-v3-turbo | RTX 4000 Ada 20GB | 5 | ~$0.16 | ~14 min | ~$0.04 |
| **Total** | | **7** | | **~14 min wall** | **~$0.08** |

Pod C breakdown (bottleneck):
- **Model load**: ~30s (one-time, turbo is smallest at ~1.5GB)
- **Per config**: ~2 min (4 tracks × ~30s each on turbo speed)
- **5 configs**: ~10 min processing
- **Overhead** (download, upload, startup): ~4 min
- **Total Pod C**: ~14 min

---

## Existing Code Reused (no modifications)

- `src/data/alignment.py` — `match_words_to_tuks()`, `load_audio_mono()`, `strip_vishram_markers()`
- `src/data/sikhnet.py` — `get_artist_tracks()`, `lookup_shabad()`, `ascii_to_unicode()`, `load_catalog()`
- `src/data/pipelines/faster_whisper.py` — Pattern for transcribe settings (not imported directly, replicated)
- `config.py` — `INITIAL_PROMPT` (Mool Mantar)
- `run_on_pod.py` — `create_pod()`, `wait_for_pod()`, `ssh_cmd()`, `scp_to_pod()`, `create_tarball()` patterns

---

## Tarball Contents

Standard project files + explicitly include:
- `database.sqlite` (151MB, needed for canonical text lookups)
- `data/manifests/test_catalog.json` (created by launcher before tarball)

Exclude: `data/audio/`, `data/segments/`, `logs/`, `.git/`, `.env`

---

## Progressive Analysis Storage

Results are stored incrementally as each config completes — no waiting for all 7.

### On-pod: per-config results written immediately

After each config finishes transcription + alignment on 4 tracks, the pod script:

1. **Pushes HF split immediately** — split is visible in dataset viewer as soon as that config is done
2. **Writes local JSON report** to `logs/{config_name}_report.json`:

```json
{
    "config_name": "v3_turbo_rep",
    "model_name": "large-v3-turbo",
    "params": {"beam_size": 3, "repetition_penalty": 1.1, "...": "..."},
    "completed_at": "2026-03-27T14:32:00Z",
    "tracks": [
        {
            "recording_id": "abc123",
            "style_bucket": "hazoori",
            "artist_name": "Bhai Harjinder Singh",
            "segments_produced": 28,
            "tuks_matched": 22,
            "tuks_total": 30,
            "tuk_coverage": 0.733,
            "mean_match_score": 0.612,
            "median_match_score": 0.645,
            "mean_confidence": 0.891,
            "repetitions_captured": 14,
            "total_duration_sec": 187.4
        }
    ],
    "aggregate": {
        "total_segments": 95,
        "mean_match_score": 0.587,
        "median_match_score": 0.621,
        "mean_confidence": 0.873,
        "tuk_coverage": 0.714,
        "total_duration_sec": 612.8
    }
}
```

3. **Writes `logs/{config_name}_COMPLETE`** marker file (polled by launcher)

### On launcher: incremental collection and comparison

The launcher polls each pod. As each config completes:

1. **SCP the report JSON** back to local `data/analysis/{config_name}_report.json`
2. **Print live summary** to terminal — shows completed configs vs pending:

```
=== Comparison Progress (4/7 complete) ===

Config             | Segments | Match Score | Confidence | Tuk Coverage
-------------------|----------|-------------|------------|-------------
large_v2           |       95 |       0.587 |      0.873 |       71.4%
large_v3           |      102 |       0.623 |      0.891 |       76.2%
v3_turbo_base      |       98 |       0.601 |      0.882 |       73.8%
v3_turbo_rep       |       91 |       0.595 |      0.879 |       72.1%
v3_turbo_temp0     |  ... running on Pod C ...
v3_turbo_beam5     |  ... pending ...
v3_turbo_nocond    |  ... pending ...
```

3. **After all 7 complete**, writes final `data/analysis/comparison_summary.json` with:
   - All 7 reports merged
   - Per-style breakdown (which config wins per style bucket)
   - Group A rankings (model comparison)
   - Group B rankings (turbo parameter impact)
   - Recommended config for full-scale with rationale

### File layout

```
data/analysis/
├── large_v2_report.json          # arrives first (~8 min)
├── large_v3_report.json          # arrives first (~8 min)
├── v3_turbo_base_report.json     # arrives ~10 min
├── v3_turbo_rep_report.json      # arrives ~10 min
├── v3_turbo_temp0_report.json    # arrives ~12 min
├── v3_turbo_beam5_report.json    # arrives ~12 min
├── v3_turbo_nocond_report.json   # arrives ~14 min
└── comparison_summary.json       # written after all 7 complete
```

---

## Verification

1. Check HF dataset viewer: `https://huggingface.co/datasets/surindersinghssj/gurbani-asr-model-comparison`
2. Verify **7 splits** exist (`large_v2`, `large_v3`, `v3_turbo_base`, `v3_turbo_rep`, `v3_turbo_temp0`, `v3_turbo_beam5`, `v3_turbo_nocond`)
3. Verify audio is playable in HF viewer
4. Spot-check `whisper_text` vs `canonical_line` for quality
5. Compare `match_score` and `avg_confidence` across all 7 configs
6. Check turbo variations: do any decoding params consistently improve scores across all 4 styles?

---

## Timeline

| Step | Time | Notes |
|------|------|-------|
| Select + download 4 tracks | ~5 min | 1 per style via SikhNet API |
| Create tarball + templates | ~3 min | One-time template creation |
| Launch 3 parallel pods | ~1 min | Script trigger |
| All 3 pods complete | ~14 min | Wall time = Pod C (5 turbo configs) |
| Review results on HF viewer | ~5 min | Compare 7 splits side-by-side |
| **Total** | **~28 min** | **~$0.08 total cost** |
