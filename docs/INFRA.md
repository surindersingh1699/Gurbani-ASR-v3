# Surt — Infrastructure

## Servers

### Hetzner CX53 (orchestrator + scraper)
- 8 vCPU, 32 GB RAM, 240 GB SSD
- ~$19/mo — billing by hour, total ~$3 for 5 days active use
- Add 200 GB attached volume (~$9/mo) if 240 GB disk fills up
- After scraping done: downgrade to CX22 ($4/mo) for cron-only use
- DO NOT run GPU work here — CPU only, orchestration only

### RunPod A100 (GPU work — alignment + training)
- Serverless pods — spin up via API, pay per second
- $1.49/hr for A100 40GB
- Pods terminate automatically after job completes
- No idle billing — cost = 0 when nothing is training

### HF Hub (storage — single source of truth)
- Dataset repo: `YOUR_HF_USERNAME/gurbani-asr-dataset` (private)
- Model repo: `YOUR_HF_USERNAME/surt-whisper-small` (private)
- Free tier: unlimited private with Pro account ($9/mo)
- Every aligned data shard pushed here
- Every training checkpoint pushed here after each phase
- Final INT8 model pushed here

## Secrets (all in Infisical)
| Key | Value | Used by |
|-----|-------|---------|
| `RUNPOD_API_KEY` | RunPod API key | orchestrator, alignment coordinator |
| `RUNPOD_ENDPOINT` | RunPod endpoint ID for training | orchestrator |
| `RUNPOD_ALIGN_ENDPOINT` | RunPod endpoint ID for alignment | alignment coordinator |
| `HF_TOKEN` | HuggingFace write token | all scripts that push to HF |
| `HF_DATASET_REPO` | `YOUR_HF_USERNAME/gurbani-asr-dataset` | scraper, train scripts |
| `HF_MODEL_REPO` | `YOUR_HF_USERNAME/surt-whisper-small` | train scripts |
| `SLACK_WEBHOOK` | Slack incoming webhook URL | notify.py |
| `APPROVAL_URL` | Cloudflare tunnel URL for approval server | notify.py |

Access in scripts:
```bash
infisical run --env=prod -- python src/pipeline/orchestrator.py
```

Or in Python:
```python
import os
hf_token = os.environ["HF_TOKEN"]  # injected by infisical run
```

## Cloudflare tunnel
Already set up on Hetzner from previous work.
Exposes approval_server.py (Flask, port 5055) to the internet.
Command:
```bash
cloudflared tunnel --url http://localhost:5055
```
Copy the generated URL → add to Infisical as `APPROVAL_URL`.

## RunPod Docker image
Both alignment and training jobs run in the same Docker image.
Base: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

Additional packages in image:
```dockerfile
RUN pip install \
  transformers>=4.40 \
  peft>=0.11 \
  datasets>=2.19 \
  huggingface-hub>=0.23 \
  faster-whisper>=1.0 \
  whisperx>=3.1 \
  ctranslate2>=4.0 \
  audiomentations>=0.36 \
  demucs>=4.0 \
  evaluate>=0.4 \
  jiwer>=3.0 \
  soundfile librosa
```

Entry point reads `input.task` from RunPod payload:
- `input.task == "align"` → runs alignment pipeline
- `input.task == "train"` → runs training pipeline
- `input.task == "quantise"` → runs Phase 5 quantisation

## Total cost breakdown

### Data pipeline
| Item | Cost |
|------|------|
| Hetzner CX53 ~5 days | ~$3 |
| RunPod A100 alignment (4 pods × 3 hrs) | ~$18 |
| HF Hub storage | Free |
| **Data pipeline total** | **~$21** |

### Training pipeline
| Phase | GPU hours | Cost |
|-------|-----------|------|
| Phase 1 text LM | ~2 hrs | ~$3 |
| Phase 2 sehaj path | ~10 hrs | ~$15 |
| Phase 3 studio kirtan | ~10 hrs | ~$15 |
| Phase 4 live kirtan | ~18 hrs | ~$27 |
| Phase 5 quantise | ~1 hr | ~$2 |
| **Training total** | **~41 hrs** | **~$62** |

### Grand total: ~$83

## GitHub Actions (CI/CD)
Deploys script changes to Hetzner automatically on push to main.
Secrets needed in GitHub repo settings:
- `HETZNER_HOST` — Hetzner server IP
- `HETZNER_USER` — SSH username
- `HETZNER_SSH_KEY` — private SSH key

Workflow at `.github/workflows/deploy.yml` copies changed Python files
to Hetzner and restarts the approval server systemd service.

## systemd services on Hetzner
```
surt-approval.service   — Flask approval server, always running on :5055
```

Cron job (not systemd):
```
*/10 * * * *   orchestrator.py — checks state.json, advances pipeline
```
