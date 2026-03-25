# config.py — Surt project settings
# All tuneable values live here. Import this everywhere.
# Secrets (API keys, tokens) come from Infisical at runtime — never hardcode here.

import os

# ─── Model ────────────────────────────────────────────────────────────────────
BASE_MODEL      = "openai/whisper-small"   # upgrade to whisper-medium if recall < 85%
LANGUAGE        = "pa"                     # Punjabi
TASK            = "transcribe"

# ─── HF Hub repos ─────────────────────────────────────────────────────────────
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "YOUR_HF_USERNAME/gurbani-asr-dataset")
HF_MODEL_REPO   = os.environ.get("HF_MODEL_REPO",   "YOUR_HF_USERNAME/surt-whisper-small")

# ─── RunPod ───────────────────────────────────────────────────────────────────
RUNPOD_ENDPOINT       = os.environ.get("RUNPOD_ENDPOINT",       "YOUR_RUNPOD_ENDPOINT_ID")
RUNPOD_ALIGN_ENDPOINT = os.environ.get("RUNPOD_ALIGN_ENDPOINT", "YOUR_RUNPOD_ALIGN_ENDPOINT_ID")

# ─── LoRA ─────────────────────────────────────────────────────────────────────
LORA_RANK       = 16
LORA_ALPHA      = 32       # 2 × rank
LORA_DROPOUT    = 0.05

# Target modules per phase — cumulative (each phase adds to previous)
LORA_TARGETS_PHASE_1 = [
    "decoder.layers.*.self_attn.q_proj",
    "decoder.layers.*.self_attn.k_proj",
    "decoder.layers.*.self_attn.v_proj",
]
LORA_TARGETS_PHASE_2 = LORA_TARGETS_PHASE_1 + [
    "decoder.layers.*.encoder_attn.q_proj",
    "decoder.layers.*.encoder_attn.k_proj",
]
LORA_TARGETS_PHASE_3_PLUS = LORA_TARGETS_PHASE_2 + [
    "encoder.layers.*.self_attn.q_proj",
    "encoder.layers.*.self_attn.v_proj",
]

def lora_targets_for_phase(phase: int) -> list[str]:
    if phase <= 1:   return LORA_TARGETS_PHASE_1
    elif phase == 2: return LORA_TARGETS_PHASE_2
    else:            return LORA_TARGETS_PHASE_3_PLUS

# ─── Training hyperparameters per phase ───────────────────────────────────────
TRAINING_CONFIG = {
    1: {"learning_rate": 1e-4, "warmup_steps": 200,  "max_steps": 2000, "batch_size": 32},
    2: {"learning_rate": 1e-4, "warmup_steps": 500,  "max_steps": 4000, "batch_size": 16},
    3: {"learning_rate": 5e-5, "warmup_steps": 300,  "max_steps": 3000, "batch_size": 16},
    4: {"learning_rate": 3e-5, "warmup_steps": 300,  "max_steps": 5000, "batch_size": 16},
    5: None,  # Phase 5 is quantisation only, no training
}

GRADIENT_ACCUMULATION_STEPS = 2   # effective batch = batch_size × 2

# ─── WER targets per phase ────────────────────────────────────────────────────
WER_TARGETS = {
    1: None,    # Phase 1: perplexity, not WER
    2: 15.0,    # < 15% on sehaj path
    3: 25.0,    # < 25% on studio kirtan
    4: 40.0,    # < 40% on live kirtan (relaxed — recall is the real metric)
}
RECALL_TARGET = 85.0  # top-3 shabad recall end-to-end

# ─── Data filtering ───────────────────────────────────────────────────────────
MIN_DURATION_SEC    = 120      # 2 min
MAX_DURATION_SEC    = 5400     # 90 min
MIN_GURMUKHI_RATIO  = 0.80     # BaniDB text must be ≥80% Gurmukhi chars
MIN_SNR_DB          = 20       # reject audio below 20dB SNR

# ─── Scraper politeness ───────────────────────────────────────────────────────
DELAY_SIKHNET    = 0.5   # seconds between SikhNet API calls
DELAY_BANIDB     = 0.3   # seconds between BaniDB calls
DELAY_DOWNLOAD   = 0.3   # seconds between audio downloads
MAX_DL_CONCURRENT   = 8  # parallel audio downloads
MAX_BANIDB_CONCURRENT = 5 # parallel BaniDB calls

# ─── Inference ────────────────────────────────────────────────────────────────
VAD_THRESHOLD       = 0.4    # silero-vad speech probability threshold
BUFFER_SECONDS      = 3.0    # audio buffer before sending to ASR
OVERLAP_SECONDS     = 0.75   # overlap between consecutive chunks (25%)
BEAM_SIZE           = 2      # speed over accuracy for live use
APPROVAL_SERVER_PORT = 5055

# Mool Mantar as initial prompt — primes decoder toward Gurbani on every call
INITIAL_PROMPT = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR         = "data"
AUDIO_DIR        = "data/audio"
MANIFEST_DIR     = "data/manifests"
CATALOG_FILE     = "data/manifests/sikhnet_catalog.json"
BANIDB_CACHE     = "data/manifests/banidb_cache.json"
CHECKPOINT_FILE  = "data/manifests/scraper_checkpoint.json"
STATE_FILE       = "state.json"
LOG_DIR          = "logs"
