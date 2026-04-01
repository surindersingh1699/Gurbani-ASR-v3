# Surt — Autonomous Training Pipeline

## Overview

The pipeline runs autonomously on Hetzner VPS using plain Python + cron.
NO Claude Code daemon. NO LLM as orchestrator. Just Python scripts + REST APIs.
Claude Code is used ON DEMAND from your laptop only when the pipeline errors.

## The semi-auto contract

- Pipeline runs fully autonomously within each phase
- Pipeline PAUSES after each phase and posts WER report to Slack
- You tap Approve or Retry on your phone
- Only then does the next phase start
- You never need to SSH in during normal operation

## Infrastructure

| Component | What | Why |
| --- | --- | --- |
| Hetzner CX53 | Orchestrator, scraper, cron | Already paid, CPU only |
| RunPod GPU | Forced alignment + training | Pay per second |
| HF Hub | Dataset + all checkpoints | Free, versioned, crash-safe |
| Infisical | Secrets (keys, tokens) | Already set up |
| Slack | Notifications + approval buttons | Already integrated |
| Cloudflare tunnel | Expose approval server | Already set up on Hetzner |

## Data preparation pipeline (CPU + GPU)

**Hetzner server (CPU):**

1. SikhNet scraping — discover artists, fetch track metadata with `shabadId`
2. STTM lookup — `shabadId` -> local `database.sqlite` -> exact canonical Gurmukhi text
3. Audio downloading — async MP3 downloads from SikhNet

**RunPod GPU:**

1. Whisper large-v3 timestamp extraction — audio + known canonical text -> tuk-level timestamps
   (NO Whisper transcription — text is already known from STTM database)

**Hetzner server (CPU):**

1. Post-processing — build HF dataset from aligned segments, push to Hub

## State machine

`state.json` is the single source of truth. Never edit manually except to fix errors.

```text
pending -> (approved) -> training -> completed -> awaiting_approval
                                      |
                                   (you approve)
                                      |
                                   pending (phase+1)
```

States:

- `pending` — waiting for approval to start this phase
- `training` — RunPod job running, orchestrator polling
- `completed` — job done, WER calculated, waiting for your review
- `awaiting_approval` — WER report sent to Slack, waiting for button tap
- `approved` — you tapped Approve, will advance on next cron tick
- `retry` — you tapped Retry, will rerun current phase on next tick
- `failed` — unrecoverable error, Claude Code intervention needed

## Cron schedule

```bash
# /etc/cron.d/surt-pipeline
*/10 * * * * surinder cd ~/surt && infisical run --env=prod -- \
  .venv/bin/python src/pipeline/orchestrator.py >> logs/pipeline.log 2>&1
```

## Approval server

Tiny Flask app running as systemd service on port 5055.
Exposed via Cloudflare tunnel (already set up on Hetzner).
Endpoints:

- `GET /approve` — advance to next phase
- `GET /retry` — rerun current phase
- `GET /skip` — skip current phase (emergency use only)
- `GET /reset` — reset to pending (after Claude Code fixes)
- `GET /status` — current state.json as JSON
- `GET /logs` — last 50 lines of pipeline.log

## Slack notification format

After each phase, you receive:

```text
Phase 3 complete
WER: 22.4% (target: 25%)
Status: Passed
[Approve — next phase] [Retry this phase] [View full logs]
```

If error:

```text
Pipeline error — Phase 3
<error message>
<last 20 log lines>
To fix: claude --dangerously-skip-permissions -p "..."
[View full logs] [Skip this phase] [Reset to pending]
```

## Error handling tiers

### Tier 1 — Auto-retry (no human needed)

- RunPod timeout or network blip
- HF upload rate limit
- 429 from any API
- Max 2 auto-retries, then escalate to Tier 2

### Tier 2 — Claude Code on laptop (you trigger manually)

- CUDA out of memory -> reduce batch_size in config.py
- HF push failed with bad token -> fix HF_TOKEN in Infisical
- Corrupted state.json -> restore from logs
- WER not improving -> check if wrong checkpoint loaded
- RunPod endpoint down -> update config

### How to trigger Claude Code repair

Copy the command from the Slack error message and run on your laptop:

```bash
claude --dangerously-skip-permissions -p \
  "Fix the Surt training pipeline. Read CLAUDE.md first. \
   SSH into Hetzner, check logs, diagnose, fix, reset state to pending."
```

## Parallel upload coordination (shard counter)

When multiple RunPod pods process data in parallel, they each need unique shard
indices so HF Hub receives sequentially named files instead of random hashes.

A lightweight Flask counter server on Hetzner (:9112) provides atomic index
allocation. Each pod claims an index before uploading.

### Flow per agent

```text
Agent starts
    │
    ├─ GET http://138.199.174.101:9112/next?agent=pod-3
    │   └─ returns {"index": 7}
    │
    ├─ Process data → local parquet
    │
    ├─ Upload as data/train-00007.parquet to HF Hub
    │
    └─ POST http://138.199.174.101:9112/done/7
        └─ confirms upload
```

### Monitoring

```bash
# Check progress from anywhere
curl http://138.199.174.101:9112/status
# Returns: {"counter": 12, "total_completed": 10, "missing": [4, 8]}
```

Missing shards = agents that claimed an index but haven't confirmed upload yet.
Could mean still uploading, or the pod died. Investigate if missing shards
persist after all pods have terminated.

### Reset between batches

```bash
curl -X POST http://138.199.174.101:9112/reset
```

Always reset before starting a new upload batch to avoid index gaps.

## Parallel forced alignment (data prep phase only)

Whisper timestamp extraction is parallelised across 4 RunPod pods simultaneously.
Each pod handles ~100 hours of audio, pushes its shard to HF independently.
Training phases 1-4 are SEQUENTIAL — cannot be parallelised (curriculum dependency).

## Full timeline estimate

| Step | Where | Time | Parallel? |
| --- | --- | --- | --- |
| SikhNet scrape | Hetzner (CPU) | ~4 hrs | 2x with split queries |
| MP3 download | Hetzner (CPU) | ~4 hrs | 8-16 concurrent |
| Whisper timestamp extraction | RunPod GPU (4 pods) | ~3 hrs | Yes — 4 pods |
| Phase 1 train | RunPod GPU | ~2 hrs | No |
| Phase 2 train | RunPod GPU | ~10 hrs | No |
| Phase 3 train | RunPod GPU | ~10 hrs | No |
| Phase 4 train | RunPod GPU | ~18 hrs | No |
| Phase 5 quant | RunPod GPU | ~1 hr | No |
| **Total** | | **~52 hrs** | **~1.5 days wall-clock** |

## Total cost estimate

| Item | Cost |
| --- | --- |
| RunPod GPU — alignment (4 pods x 3 hrs) | ~$18 |
| RunPod GPU — training all phases (~41 hrs) | ~$61 |
| Hetzner CX53 (5 days) | ~$3 |
| HF Hub storage | Free |
| **Total** | **~$82** |
