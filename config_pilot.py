"""Pilot training config — overrides config.py for Phase 1+2 pilot run.

Usage:
    python src/model/train.py --phase 1 --config config_pilot
    python src/model/train.py --phase 2 --config config_pilot
"""
from config import *  # noqa: F401,F403 — inherit all production values

# ─── Pilot subset ────────────────────────────────────────────────────────────
PILOT_SUBSET_SIZE = 5000        # rows to sample (per phase)

# ─── HF repos ────────────────────────────────────────────────────────────────
# Separate pilot model repo so we don't pollute production
HF_MODEL_REPO = "surindersinghssj/surt-whisper-small-pilot"

# ─── Training overrides ──────────────────────────────────────────────────────
TRAINING_CONFIG = {
    1: {  # Phase 1: text LM priming (reduced for pilot)
        "learning_rate": 1e-4,
        "warmup_steps": 100,
        "max_steps": 1000,     # half of full 2000
        "batch_size": 32,
        "save_steps": 200,
        "eval_steps": 200,
    },
    2: {  # Phase 2: sehaj path audio (reduced for pilot)
        "learning_rate": 1e-4,
        "warmup_steps": 50,
        "max_steps": 500,
        "batch_size": 16,
        "save_steps": 100,
        "eval_steps": 100,
    },
    3: None,  # not used in pilot
    4: None,  # not used in pilot
    5: None,  # quantisation only
}
