# Surt — 5-Phase Training Plan

## Why curriculum learning

Gurbani is a composite of Old Punjabi, Braj Bhasha, Sanskrit, Persian, and Arabic.
Whisper has almost never seen this vocabulary. Training on noisy kirtan from the start
causes the model to learn noise patterns before it knows the vocabulary — WER never
recovers. Curriculum: easy -> hard, text -> clean audio -> noisy audio.

## Data sources

- **Sehaj path**: clean recitation, no instruments, 90 hours (sourced separately)
- **Kirtan audio**: SikhNet tracks with shabadId -> STTM database ground truth labels, ~400 hours after filtering
  - Phase 3 (hazoori/puratan): `style_bucket in ["hazoori", "puratan"]`
  - Phase 4 (live/noisy): `style_bucket in ["akj", "taksali", "live", "mixed"]`
- **Text corpus**: STTM full SGGS + Dasam Granth + Bhai Gurdas Ji Vara + Nitnem + Sundar Gutka in Unicode Gurmukhi

All data lives on HF Hub: `surindersinghssj/gurbani-asr-dataset`
Filter by `phase` field on each record (set by scraper during catalog build).

## Data preparation — timestamp extraction + canonical text matching

**There is NO Whisper transcription used as training labels.** Every SikhNet track
has a `shabadId` that maps to exact canonical text in our local `database.sqlite`.
We use Whisper large-v2 with `word_timestamps=True` as a timestamp oracle (NOT
WhisperX — no Punjabi alignment model), then match the timestamped words to
canonical tuks via matra-normalised F1 scoring.

### Vishram-aware alignment

56% of STTM lines have a primary vishram (`;`) marking the natural pause point.
Kirtanis often sing each half separately. The alignment pipeline expands each line
into [full, first_half, second_half] match targets and captures whichever pattern
the kirtani uses. Combined with repetition detection, a single pangti can yield
4-6 training segments (2 halves × 2-3 repetitions each).

Pipeline: audio + canonical text → Whisper timestamps → vishram-aware matching → segments → HF Hub

## HF Hub checkpoint naming

Each phase saves to a separate folder in the model repo:

- `surindersinghssj/surt-whisper-small/phase1/`
- `surindersinghssj/surt-whisper-small/phase2/`
- `surindersinghssj/surt-whisper-small/phase3/`
- `surindersinghssj/surt-whisper-small/phase4/`
- `surindersinghssj/surt-whisper-small/final-float/` — merged float32, reversible source
- `surindersinghssj/surt-whisper-small/final/` — INT8, deploy-ready

Each phase loads from the previous phase's HF checkpoint.
NEVER start a phase from scratch — always resume from previous checkpoint.

---

## Phase 1 — Text language model priming

**Goal**: Teach the decoder Gurbani vocabulary before hearing any audio.
**Why first**: Highest ROI step. A few GPU hours, no audio needed.
The decoder's self-attention learns Gurbani word sequences and stops preferring
modern Punjabi. WER improvement in all subsequent phases is significant.

**Data**: STTM full SGGS text + Nitnem + Sundar Gutka (Unicode Gurmukhi) + Dasam Granth + Bhai Gurdas Vara
Weight Nitnem heavily — these are the most common kirtan shabads.

**LoRA targets**: Decoder self-attention only (Q, K, V)
**Rank**: 16

**Training config**:

```python
Seq2SeqTrainingArguments(
    learning_rate=1e-4,
    warmup_steps=200,
    max_steps=2000,
    per_device_train_batch_size=32,  # text is cheap, large batch fine
    gradient_accumulation_steps=2,
    fp16=True,
    eval_steps=200,
    save_steps=200,
    save_total_limit=3,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="wandb",
    metric_for_best_model="loss",    # WER not applicable for text-only
)
```

**Success metric**: Perplexity drop ~30-50% vs untrained decoder on Gurbani text.
**Expected GPU time**: ~1-2 hours.
**Checkpoint**: save to `phase1/` on HF Hub before terminating pod.

---

## Phase 2 — Clean audio (sehaj path)

**Goal**: Learn acoustic-to-Gurmukhi mapping with zero noise.
Establishes the baseline acoustic model before any instrument interference.

**Data**: Sehaj path recordings only. NO kirtan, NO instruments.

- Single reader, clean mic, no background noise
- SNR > 20dB required (enforced by data prep pipeline)
- Reject anything with harmonium, tabla, or background crowd

**Forced alignment**: Sehaj path aligns cleanly — Whisper large-v3 timestamp
extraction handles recitation well. Alignment confidence should be high. The
canonical text comes from STTM database (known in advance), NOT from Whisper
transcription.

**LoRA targets**: Decoder self-attn (Q, K, V) + cross-attn (Q, K)
**Rank**: 16 self-attn, 8 cross-attn

**Training config**:

```python
Seq2SeqTrainingArguments(
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=4000,           # ~2 epochs on 90hr dataset
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,  # effective batch = 32
    fp16=True,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="wandb",
    metric_for_best_model="wer",
    greater_is_better=False,
)
```

**Data loading**:

```python
dataset = load_dataset(
    "surindersinghssj/gurbani-asr-dataset",
    split="train",
).filter(lambda x: x["phase"] == 2).shuffle(seed=42)
```

> **Do NOT use `streaming=True`.** Streaming limits shuffle to a 1000-row buffer,
> causing clustering by artist/raag. Download the filtered dataset once — it fits
> in memory for Phase 2 (~90 hours, ~150K rows, ~4GB). Full shuffle is critical
> for training convergence.

**Success metric**: WER < 15% on held-out sehaj path test set.
**Expected GPU time**: ~8-12 hours.
**Checkpoint**: save to `phase2/` on HF Hub. Push every 500 steps as backup.

---

## Phase 3 — Studio kirtan (hazoori + puratan)

**Goal**: Add instrument awareness. Model learns vocal line exists above harmonium.
Train on mixed audio directly — the model must learn to handle instruments from the start.

**Data**: `style_bucket in ["hazoori", "puratan"]` from SikhNet dataset.
~75-100 hours. Professional recordings, controlled acoustic environments.
Train on original mixed audio (NOT Demucs-separated).

**No Demucs by default.** Demucs vocal separation is an optional experiment to try
later if Phase 3 WER is unacceptably high on music-heavy files. It adds GPU cost
and pipeline complexity for uncertain benefit at this stage.

**LoRA targets**: All previous + encoder self-attn (Q, V)
**Rank**: 16 decoder self-attn, 8 cross-attn, 8 encoder self-attn

**Training config**:

```python
Seq2SeqTrainingArguments(
    learning_rate=5e-5,       # lower LR — don't overwrite Phase 2
    warmup_steps=300,
    max_steps=3000,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=True,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="wandb",
)
```

**Data augmentation** (applied online during training, not pre-processed):

```python
augment = Compose([
    PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
    # No RIR or noise yet — that's Phase 4
])
```

**Success metric**: WER < 25% on studio kirtan test set.
**Expected GPU time**: ~8-12 hours.
**Checkpoint**: save to `phase3/` on HF Hub.

---

## Phase 4 — Live kirtan (AKJ, taksali, mixed, live gurdwara)

**Goal**: Handle real-world conditions. Hall reverb, sangat, varying mic quality.
This is the hardest phase. WER target is relaxed because shabad lookup
recall (not WER) is the real metric.

**Data**: `style_bucket in ["akj", "taksali", "live", "mixed"]` from SikhNet.
~200-250 hours. Mixed audio (NOT Demucs separated — use original recordings).

**Style-specific challenges**:

- AKJ: fast tempo, repetitive loops, sangat joins in — train loop boundary awareness
- Taksali: slow, precise santhya pronunciation — unique phoneme articulations
- Live gurdwara: hall reverb varies by room size — use RIR augmentation
- Rag kirtan: melismatic vocals (one syllable over many notes) — CTC handles this

**Data augmentation** (all applied online):

```python
augment = Compose([
    AddBackgroundNoise(
        sounds_path="./harmonium_samples",
        min_snr_db=5, max_snr_db=20,
        p=0.6
    ),
    RoomSimulator(
        min_size_x=8, max_size_x=40,   # small room to large darbar
        min_size_y=8, max_size_y=40,
        min_size_z=3, max_size_z=12,
        p=0.5
    ),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.4),
])
```

**Training config**:

```python
Seq2SeqTrainingArguments(
    learning_rate=3e-5,       # lower still — careful not to overwrite Phase 3
    warmup_steps=300,
    max_steps=5000,           # more steps for harder data
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    fp16=True,
    eval_steps=500,
    save_steps=500,
    save_total_limit=3,
    gradient_checkpointing=True,
    dataloader_num_workers=4,
    report_to="wandb",
)
```

**Success metric**:

- WER < 40% on live kirtan test set (relaxed — noisy audio is hard)
- Top-3 shabad recall > 85% end-to-end (this is the real target)

**Expected GPU time**: ~15-20 hours.
**Checkpoint**: save to `phase4/` on HF Hub.

---

## Phase 5 — Quantisation and packaging

**Goal**: Convert Phase 4 LoRA model to deployment-ready INT8 faster-whisper.
No training in this phase — pure conversion and validation.

**INT8 is a non-destructive export.** The merged float32 model is always preserved
on HF Hub (`phase4/` checkpoint + `final-float/`). If INT8 degrades recall, we can
re-export with float16 or adjust quantisation settings without retraining.

**Steps**:

1. Merge LoRA adapter into base weights → save as `final-float/` on HF Hub
2. Convert to CTranslate2 INT8 format for faster-whisper → save as `final/`
3. Validate: run full inference pipeline, check shabad recall still > 85%
4. If recall drops below threshold, fall back to float16 (`compute_type="float16"`)
5. Package with vocab constraint files (suppress_tokens list, STTM wordset)
6. Push both `final-float/` (reversible source) and `final/` (INT8 deploy) to HF Hub

```python
# Step 1: merge LoRA → float model (ALWAYS keep this)
model = model.merge_and_unload()
model.save_pretrained("./merged")
# Push ./merged to HF Hub as final-float/ — this is the reversible source

# Step 2: convert to CTranslate2 INT8
# ct2-opus-mt-convert --model ./merged --output_dir ./surt-int8 --quantization int8

# Step 3: validate INT8
from faster_whisper import WhisperModel
model = WhisperModel("./surt-int8", device="cpu", compute_type="int8")
# Run eval on test set, confirm recall > 85%

# Step 4: if INT8 recall drops, re-export as float16 instead
# ct2-opus-mt-convert --model ./merged --output_dir ./surt-f16 --quantization float16
```

**Expected GPU time**: ~1-2 hours.
**Output**: `final-float/` (merged float, reversible) + `final/` (INT8 deploy) on HF Hub.

---

## WER targets summary

| Phase | Test set | WER target | If missed |
| --- | --- | --- | --- |
| 1 | Gurbani text perplexity | -30% vs baseline | Check STTM data loading |
| 2 | Sehaj path | < 15% | More data or check alignment |
| 3 | Studio kirtan | < 25% | Try Demucs separation as experiment |
| 4 | Live kirtan | < 40% | Expected — focus on recall not WER |
| 4 | End-to-end shabad recall | > 85% top-3 | Tune BM25 layer first |

## Dataset split

- Train: 85%
- Validation: 10%
- Test: 5%

Stratify by style_bucket — each style must appear in test set.
Keep test set fixed across all phases for fair comparison.

## GPU utilisation and dataloader workers

Training GPU utilisation will be ~60-70% without tuning. The bottleneck is CPU-side
preprocessing: loading audio files, computing mel spectrograms, and (in Phases 3-4)
online augmentation (PitchShift, RoomSimulator). With `dataloader_num_workers=0`
(the default), the main thread does all preprocessing while the GPU idles.

**Mitigations (all included in training config above):**

1. `dataloader_num_workers=4` — overlaps CPU preprocessing with GPU forward/backward.
   4 workers is a good default for RunPod pods (8 vCPUs typical). Increase to 6-8
   if using a pod with 16+ vCPUs.

2. `gradient_checkpointing=True` — reduces VRAM usage by ~60%, allowing larger
   effective batch sizes. Trades ~20% training speed for much better VRAM headroom.
   Essential safety net for OOM prevention on 24GB GPUs.

3. For Phase 2 (clean audio, no augmentation): consider pre-computing mel
   spectrograms offline and saving to HF dataset as a `mel` column. This eliminates
   the CPU bottleneck entirely for the longest training phase:

   ```python
   from transformers import WhisperFeatureExtractor
   fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
   dataset = dataset.map(
       lambda x: {"mel": fe(x["audio"]["array"], sampling_rate=16000).input_features[0]}
   )
   ```

4. Phases 3-4 add online augmentation (PitchShift, RoomSimulator) which is
   CPU-intensive. The 4 workers help here, but if GPU util drops below 50%,
   increase workers or pre-augment a portion of the dataset offline.

## Training runs on RunPod

Each phase = one RunPod job submitted by the Hetzner orchestrator.
Train script downloads dataset from HF Hub (cached), pushes checkpoint to HF every
`save_steps`. Pod terminates immediately after final push — no idle billing.
Orchestrator waits for completion, posts WER to Slack, waits for your approval.

Set `WANDB_PROJECT=surt` and `WANDB_API_KEY` environment variables on the RunPod pod.
W&B logs training loss, eval WER, learning rate, and GPU utilisation per step.
Use the W&B dashboard to monitor training across phases and compare runs.
