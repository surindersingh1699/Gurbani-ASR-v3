# Surt — 5-Phase Training Plan

## Why curriculum learning
Gurbani is a composite of Old Punjabi, Braj Bhasha, Sanskrit, Persian, and Arabic.
Whisper has almost never seen this vocabulary. Training on noisy kirtan from the start
causes the model to learn noise patterns before it knows the vocabulary — WER never
recovers. Curriculum: easy → hard, text → clean audio → noisy audio.

## Data sources
- **Sehaj path**: clean recitation, no instruments, 90 hours (sourced separately)
- **Kirtan audio**: SikhNet + BaniDB ground truth labels, ~400 hours after filtering
  - Phase 3 (hazoori/puratan): `style_bucket in ["hazoori", "puratan"]`
  - Phase 4 (live/noisy): `style_bucket in ["akj", "taksali", "live", "mixed"]`
- **Text corpus**: STTM/SikhiToTheMax full SGGS + Nitnem in Unicode Gurmukhi

All data lives on HF Hub: `YOUR_HF_USERNAME/gurbani-asr-dataset`
Filter by `phase` field on each record (set by scraper during catalog build).

## HF Hub checkpoint naming
Each phase saves to a separate folder in the model repo:
- `YOUR_HF_USERNAME/surt-whisper-small/phase1/`
- `YOUR_HF_USERNAME/surt-whisper-small/phase2/`
- `YOUR_HF_USERNAME/surt-whisper-small/phase3/`
- `YOUR_HF_USERNAME/surt-whisper-small/phase4/`
- `YOUR_HF_USERNAME/surt-whisper-small/final/`  ← INT8, deploy-ready

Each phase loads from the previous phase's HF checkpoint.
NEVER start a phase from scratch — always resume from previous checkpoint.

---

## Phase 1 — Text language model priming
**Goal**: Teach the decoder Gurbani vocabulary before hearing any audio.
**Why first**: Highest ROI step. A few GPU hours, no audio needed.
The decoder's self-attention learns Gurbani word sequences and stops preferring
modern Punjabi. WER improvement in all subsequent phases is significant.

**Data**: STTM full SGGS text + Nitnem + Sundar Gutka (Unicode Gurmukhi)
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
    fp16=True,
    eval_steps=200,
    metric_for_best_model="loss",    # WER not applicable for text-only
)
```

**Success metric**: Perplexity drop ~30-50% vs untrained decoder on Gurbani text.
**Expected GPU time**: ~1-2 hours on A100.
**Checkpoint**: save to `phase1/` on HF Hub before terminating pod.

---

## Phase 2 — Clean audio (sehaj path)
**Goal**: Learn acoustic-to-Gurmukhi mapping with zero noise.
Establishes the baseline acoustic model before any instrument interference.

**Data**: Sehaj path recordings only. NO kirtan, NO instruments.
- Single reader, clean mic, no background noise
- SNR > 20dB required (enforced by data prep pipeline)
- Reject anything with harmonium, tabla, or background crowd

**whisperX alignment**: Sehaj path aligns cleanly — Whisper's baseline Punjabi
model handles recitation well. Alignment confidence should be high.

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
    metric_for_best_model="wer",
    greater_is_better=False,
)
```

**Data loading**:
```python
dataset = load_dataset(
    "YOUR_HF_USERNAME/gurbani-asr-dataset",
    split="train", streaming=True
).filter(lambda x: x["phase"] == 2)
```

**Success metric**: WER < 15% on held-out sehaj path test set.
**Expected GPU time**: ~8-12 hours on A100.
**Checkpoint**: save to `phase2/` on HF Hub. Push every 500 steps as backup.

---

## Phase 3 — Studio kirtan (hazoori + puratan)
**Goal**: Add instrument awareness. Model learns vocal line exists above harmonium.
Uses Demucs vocal separation on training data to ease the transition.

**Data**: `style_bucket in ["hazoori", "puratan"]` from SikhNet dataset.
~75-100 hours. Professional recordings, controlled acoustic environments.

**IMPORTANT — Demucs pre-processing**:
Run Demucs source separation on ALL Phase 3 audio before training.
Train on the isolated vocal stem, NOT the mixed audio.
This is a stepping stone — Phase 4 trains on mixed audio.

```bash
python -m demucs --two-stems=vocals path/to/kirtan.wav
# Use: kirtan/vocals.wav for Phase 3 training
# Save: original mixed file — needed for Phase 4
```

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
**Expected GPU time**: ~8-12 hours on A100.
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
)
```

**Success metric**:
- WER < 40% on live kirtan test set (relaxed — noisy audio is hard)
- Top-3 shabad recall > 85% end-to-end (this is the real target)
**Expected GPU time**: ~15-20 hours on A100.
**Checkpoint**: save to `phase4/` on HF Hub.

---

## Phase 5 — Quantisation and packaging
**Goal**: Convert Phase 4 LoRA model to deployment-ready INT8 faster-whisper.
No training in this phase — pure conversion and validation.

**Steps**:
1. Merge LoRA adapter into base weights
2. Convert to CTranslate2 INT8 format for faster-whisper
3. Validate: run full inference pipeline, check shabad recall still > 85%
4. Package with vocab constraint files (suppress_tokens list, STTM wordset)
5. Push final model to HF Hub as `final/`

```python
# Step 1: merge LoRA
model = model.merge_and_unload()
model.save_pretrained("./merged")

# Step 2: convert to CTranslate2
# Run from shell:
# ct2-opus-mt-convert --model ./merged --output_dir ./surt-int8 --quantization int8

# Step 3: validate
from faster_whisper import WhisperModel
model = WhisperModel("./surt-int8", device="cpu", compute_type="int8")
# Run eval on test set, confirm recall > 85%
```

**Expected GPU time**: ~1-2 hours on A100.
**Output**: `final/` directory on HF Hub, ready to `pip install` and run.

---

## WER targets summary

| Phase | Test set | WER target | If missed |
|-------|----------|------------|-----------|
| 1 | Gurbani text perplexity | -30% vs baseline | Check STTM data loading |
| 2 | Sehaj path | < 15% | More data or check alignment |
| 3 | Studio kirtan | < 25% | Check Demucs separation quality |
| 4 | Live kirtan | < 40% | Expected — focus on recall not WER |
| 4 | End-to-end shabad recall | > 85% top-3 | Tune BM25 layer first |

## Dataset split
- Train: 85%
- Validation: 10%
- Test: 5%
Stratify by style_bucket — each style must appear in test set.
Keep test set fixed across all phases for fair comparison.

## Training runs on RunPod
Each phase = one RunPod job submitted by the Hetzner orchestrator.
Train script reads from HF Hub (streaming), pushes checkpoint to HF on completion.
Pod terminates immediately after push — no idle billing.
Orchestrator waits for completion, posts WER to Slack, waits for your approval.
