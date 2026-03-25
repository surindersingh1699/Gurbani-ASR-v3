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
|-----------|------|-----|
| Hetzner CX53 | Orchestrator, scraper, cron | Already paid, CPU only |
| RunPod A100 | GPU alignment + training | $1.49/hr, pay per second |
| HF Hub | Dataset + all checkpoints | Free, versioned, crash-safe |
| Infisical | Secrets (keys, tokens) | Already set up |
| Slack | Notifications + approval buttons | Already integrated |
| Cloudflare tunnel | Expose approval server | Already set up on Hetzner |

## State machine
`state.json` is the single source of truth. Never edit manually except to fix errors.

```
pending → (approved) → training → completed → awaiting_approval
                                      ↓
                                   (you approve)
                                      ↓
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
```
✅ Phase 3 complete
WER: 22.4% (target: 25%)
Status: Passed
[Approve — next phase] [Retry this phase] [View full logs]
```

If error:
```
🚨 Pipeline error — Phase 3
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
- CUDA out of memory → reduce batch_size in config.py
- HF push failed with bad token → fix HF_TOKEN in Infisical
- Corrupted state.json → restore from logs
- WER not improving → check if wrong checkpoint loaded
- RunPod endpoint down → update RUNPOD_ENDPOINT in config.py

### How to trigger Claude Code repair
Copy the command from the Slack error message and run on your laptop:
```bash
claude --dangerously-skip-permissions -p \
  "Fix the Surt training pipeline. Read CLAUDE.md first. \
   SSH into Hetzner, check logs, diagnose, fix, reset state to pending."
```

## Parallel alignment (data prep phase only)
WhisperX alignment is parallelised across 4 RunPod pods simultaneously.
Each pod handles ~100 hours of audio, pushes its shard to HF independently.
Training phases 1-4 are SEQUENTIAL — cannot be parallelised (curriculum dependency).

## Full timeline estimate
| Step | Where | Time | Parallel? |
|------|-------|------|-----------|
| SikhNet scrape | Hetzner | ~4 hrs | 2× with split queries |
| MP3 download | Hetzner | ~4 hrs | 8-16 concurrent |
| whisperX align | RunPod (4 pods) | ~3 hrs | Yes — 4 pods |
| Phase 1 train | RunPod | ~2 hrs | No |
| Phase 2 train | RunPod | ~10 hrs | No |
| Phase 3 train | RunPod | ~10 hrs | No |
| Phase 4 train | RunPod | ~18 hrs | No |
| Phase 5 quant | RunPod | ~1 hr | No |
| **Total** | | **~52 hrs** | **~1.5 days wall-clock** |

## Total cost estimate
| Item | Cost |
|------|------|
| RunPod A100 — alignment (4 pods × 3 hrs) | ~$18 |
| RunPod A100 — training all phases (~41 hrs) | ~$61 |
| Hetzner CX53 (5 days) | ~$3 |
| HF Hub storage | Free |
| **Total** | **~$82** |
