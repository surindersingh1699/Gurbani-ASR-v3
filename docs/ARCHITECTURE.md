# Surt — Architecture

## Model choice
Base model: `openai/whisper-small` (244M params, 12 encoder + 12 decoder layers)
Fine-tuning: LoRA via Hugging Face PEFT
Inference: `faster-whisper` with INT8 quantisation

Upgrade path: if top-3 shabad recall < 85% on live kirtan after full training,
switch to `openai/whisper-medium` (769M params, 24 layers). Code change is one string.
Do NOT start with medium — validate pipeline on small first.

Why not base or tiny: encoder too shallow (6 layers) to separate vocals from
harmonium + tabla in live gurdwara conditions. Cannot be fixed by fine-tuning.

## Whisper architecture — what each part does

### Encoder ("the ear")
- Input: log-mel spectrogram of audio (80 mel bins, 3000 time frames = 30s)
- Output: sequence of dense vectors, one per ~20ms of audio
- Job: convert raw sound into acoustic representations
- For Surt: must learn to separate vocal line from harmonium/tabla/reverb
- LoRA targets in encoder: `self_attn.q_proj`, `self_attn.v_proj` (added Phase 3+)

### Decoder ("the language model")
Three attention mechanisms per layer:

1. **Self-attention (causal)** — looks at previously generated tokens
   - This is where Gurbani vocabulary priors live
   - PRIMARY LoRA target: rank 16 on Q, K, V projections
   - Fine-tune from Phase 1 (text-only) onward

2. **Cross-attention (encoder-decoder)** — bridges audio vectors to text tokens
   - Learns "which audio features correspond to which Gurbani word"
   - Learns to attend to vocal frequencies, ignore harmonium
   - LoRA target: rank 8 on Q, K projections
   - Fine-tune from Phase 2 onward

3. **Feed-forward network** — dense knowledge storage
   - Do NOT fine-tune — insufficient data at this scale

## LoRA configuration
```python
LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,                    # rank for decoder self-attn
    lora_alpha=32,           # scaling = 2 × rank
    lora_dropout=0.05,
    target_modules=[
        # Phase 1-2: decoder self-attention only
        "decoder.layers.*.self_attn.q_proj",
        "decoder.layers.*.self_attn.k_proj",
        "decoder.layers.*.self_attn.v_proj",
        # Phase 2+: cross-attention
        "decoder.layers.*.encoder_attn.q_proj",
        "decoder.layers.*.encoder_attn.k_proj",
        # Phase 3+: encoder self-attention (added for kirtan acoustics)
        "encoder.layers.*.self_attn.q_proj",
        "encoder.layers.*.self_attn.v_proj",
    ],
    bias="none",
)
# Trainable params: ~10-15M out of 244M total (whisper-small)
# Runs on single A100 40GB with batch_size=16
```

## The 3-layer Gurbani-only output constraint

This is the most important architectural decision. The model CANNOT output
non-Gurbani words. Three enforcement layers work together:

### Layer 1 — Token suppression (decoder, at beam search)
Suppress all token IDs whose string is not in Gurmukhi Unicode block (U+0A00–U+0A7F).
Passed as `suppress_tokens` list to faster-whisper at inference time.
~48,000 of ~50,000 Whisper vocabulary tokens are suppressed.
Only ~2,000 Gurmukhi tokens remain eligible.

```python
suppress_tokens = [
    token_id for token_str, token_id in tokenizer.get_vocab().items()
    if not is_gurmukhi_or_special(token_str)
]
```

### Layer 2 — STTM vocabulary constraint (LogitsProcessor)
Further restrict to only words that appear in the STTM/BaniDB database.
Applied as a custom LogitsProcessor during generation.
~800-1200 valid Gurbani word-tokens after this filter.
Eliminates valid Gurmukhi Unicode that is not Gurbani (e.g. modern Punjabi words).

```python
class GurbaniOnlyLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids, scores):
        mask = torch.ones(scores.shape[-1], dtype=torch.bool)
        mask[self.allowed_token_ids] = False
        scores[:, mask] = float("-inf")
        return scores
```

### Layer 3 — Post-filter (output string)
After generation, strip any token not in STTM word set.
Last line of defence — catches edge cases.
If output is empty (silence, tabla only, announcements): return None.
UI holds last confirmed shabad match on empty output — does not clear screen.

```python
def clean_to_gurbani_only(text: str, sttm_words: set) -> str:
    matraan = "ਾਿੀੁੂੇੈੋੌੰੱੲੳ਼"
    words = text.strip().split()
    return " ".join(
        w.strip("।॥,.?! ") for w in words
        if w.strip("।॥,.?! ") in sttm_words
    )
```

## Inference pipeline (live, real-time)

```
Mic → VAD chunk (silero-vad) → faster-whisper → Layer 3 filter → BM25 lookup → display
```

### Timing targets
- Audio capture → transcript: < 1.5 sec (whisper-small INT8 on CPU)
- BM25 lookup: < 10ms
- Total mic-to-display latency: < 2 seconds

### Key parameters
```python
model.transcribe(
    audio_chunk,
    language="pa",
    beam_size=2,                   # speed over accuracy for live use
    temperature=0,                  # deterministic
    suppress_tokens=suppress_tokens, # Layer 1
    initial_prompt="ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ",  # primes LM decoder
    vad_filter=True,
    without_timestamps=True,
)
```

### VAD settings
- Library: silero-vad
- Buffer: 3 seconds of speech before sending to ASR
- Overlap: 25% (0.75s) between consecutive chunks to avoid cutting tuks
- Threshold: 0.4 (speech probability)

## BM25 shabad lookup

### Matraa normalisation (most impactful single improvement)
Strip vowel diacritics from BOTH the ASR output AND the STTM index before matching.
Most ASR errors on Gurmukhi are wrong matraa with correct consonant skeleton.
This single normalisation improves top-1 recall by ~15-20%.

```python
def normalise_gurmukhi(text: str) -> str:
    matraan = "ਾਿੀੁੂੇੈੋੌੰੱੲੳ਼"
    return "".join(c for c in text if c not in matraan).strip()
```

### Matching strategy
1. BM25 on normalised text (primary, < 10ms)
2. If top-1 confidence < 0.4, try dense vector search with multilingual-e5-small
3. Show top-3 candidates with confidence scores
4. Partial transcript matching — do NOT wait for full shabad before querying

### Display behaviour
- Show top-3 matches with ang number and confidence bar
- Hold last confirmed match on screen during silence/tabla-only sections
- Never clear screen to blank — always show something
