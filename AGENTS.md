# Surt — Gurbani ASR Project

## What this project is
Surt is a real-time Gurbani speech recognition model for live kirtan search during
gurdwara diwan. It listens to live kirtan audio and identifies the shabad being sung,
displaying the matching lines from SikhiToTheMax on screen in real time.

The name comes from *surat* — the consciousness that attunes itself to Gurbani.

## The one rule that defines everything
**This model outputs ONLY Gurbani words. No Punjabi. No English. Nothing else.**
This is enforced at three layers:
1. Token suppression — non-Gurmukhi tokens get -inf logit at beam search
2. Vocabulary constraint — only STTM words allowed via LogitsProcessor
3. Post-filter — any non-Gurbani token in output is silently stripped

Read `docs/ARCHITECTURE.md` for the full technical design.
Read `docs/TRAINING.md` for the 5-phase training plan.
Read `docs/PIPELINE.md` for the autonomous training orchestration.
Read `docs/SCRAPER.md` for the SikhNet data collection + forced alignment spec.
Read `docs/INFRA.md` for infrastructure and cost details.

## Repository structure
```
surt/
├── AGENTS.md                  ← you are here
├── docs/
│   ├── ARCHITECTURE.md        ← model design, vocab constraint, inference
│   ├── TRAINING.md            ← 5-phase curriculum training plan
│   ├── PIPELINE.md            ← autonomous orchestration on Hetzner
│   ├── SCRAPER.md             ← SikhNet + STTM data collection
│   └── INFRA.md               ← servers, cost, HF Hub storage
├── src/
│   ├── data/
│   │   ├── sikhnet.py         ← SikhNet scraper (full implementation exists)
│   │   └── prepare_pilot.py   ← Whisper timestamp alignment + segmentation
│   ├── model/
│   │   ├── vocab_constraint.py ← Gurbani-only token enforcement
│   │   ├── train.py           ← LoRA fine-tuning loop (all phases)
│   │   └── quantise.py        ← Phase 5 INT8 export for faster-whisper
│   ├── inference/
│   │   ├── live.py            ← real-time VAD + transcription pipeline
│   │   └── bm25_index.py      ← STTM shabad lookup with matraa normalisation
│   └── pipeline/
│       ├── orchestrator.py    ← Hetzner cron orchestrator
│       ├── approval_server.py ← Flask approval server for Slack buttons
│       ├── runpod_client.py   ← RunPod API client
│       ├── notify.py          ← Slack notifications
│       └── hf_client.py       ← HF Hub checkpoint push/pull
├── state.json                 ← pipeline state (phase, status, wer, job_id)
├── config.py                  ← all tuneable settings
└── requirements.txt

```

## Key decisions already made — do not revisit these
- **Model**: whisper-small to start, upgrade to medium only if shabad recall < 85%
- **Fine-tuning**: LoRA only (not full fine-tune). Rank 16 decoder self-attn, rank 8 cross-attn
- **Output constraint**: Gurbani-only via 3-layer enforcement (see ARCHITECTURE.md)
- **Data source**: SikhNet tracks with shabadId → local STTM `database.sqlite` for ground truth text
- **Data prep**: Whisper large-v3 as timestamp oracle (`word_timestamps=True`) + vishram-aware matching. NO Whisper transcription as training labels. Every track has a shabadId → exact canonical text from STTM database. Timestamps are matched to canonical tuks via matra-normalised F1 scoring.
- **Database**: Local `database.sqlite` (STTM database, 151MB). NO BaniDB API calls.
- **Storage**: HF Hub is single source of truth for dataset + all checkpoints
- **Orchestration**: Hetzner VPS (plain Python cron, no LLM daemon)
- **GPU**: RunPod for both alignment and training
- **Inference runtime**: faster-whisper with INT8 quantisation (reversible — merged float model always kept on HF Hub)
- **Demucs**: NOT used by default. Optional experiment only for difficult music-heavy files if Phase 3 WER is too high
- **Shabad lookup**: BM25 with Gurmukhi matraa normalisation

## What "done" looks like
- Live audio in → Gurbani text out in < 2 seconds
- Top-3 shabad recall > 85% on live gurdwara kirtan
- Model runs on a standard laptop (CPU, INT8 or float16 fallback)
- Single command to start: `python src/inference/live.py`
