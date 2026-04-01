# Whisper Model Comparison v2 — Style-Diverse Kirtan Analysis
Generated: 2026-03-28T05:47:16.992337+00:00
25 tracks across 5 style buckets (hazoori, puratan, akj, live, mixed)

| Config | Model | Segs | Match | Confidence | Tuk Coverage |
|--------|-------|------|-------|------------|--------------|
| large_v2 | faster-whisper-large-v2 | 272 | 0.312 | 0.735 | 24.6% |
| large_v3 | faster-whisper-large-v3 | 532 | 0.342 | 0.902 | 34.5% |
| v3_turbo_base | faster-whisper-large-v3-turbo-ct2 | 310 | 0.279 | 0.824 | 35.2% |
| v3_turbo_rep | faster-whisper-large-v3-turbo-ct2 | 332 | 0.312 | 0.802 | 34.9% |
| v3_turbo_temp0 | faster-whisper-large-v3-turbo-ct2 | 431 | 0.283 | 0.840 | 13.0% |
| v3_turbo_beam5 | faster-whisper-large-v3-turbo-ct2 | 275 | 0.293 | 0.792 | 34.5% |
| v3_turbo_nocond | faster-whisper-large-v3-turbo-ct2 | 238 | 0.276 | 0.823 | 27.5% |

## Rankings
- **Match**: large_v3 > large_v2 > v3_turbo_rep > v3_turbo_beam5 > v3_turbo_temp0 > v3_turbo_base > v3_turbo_nocond
- **Confidence**: large_v3 > v3_turbo_temp0 > v3_turbo_base > v3_turbo_nocond > v3_turbo_rep > v3_turbo_beam5 > large_v2
- **Coverage**: v3_turbo_base > v3_turbo_rep > large_v3 > v3_turbo_beam5 > v3_turbo_nocond > large_v2 > v3_turbo_temp0

## Per-Bucket Coverage
| Config | hazoori | puratan | akj | live | mixed |
|--------|---------|---------|-----|------|-------|
| large_v2 | 14% | 15% | 24% | 37% | 35% |
| large_v3 | 25% | 37% | 41% | 27% | 37% |
| v3_turbo_base | 17% | 46% | 23% | 40% | 42% |
| v3_turbo_rep | 22% | 29% | 30% | 44% | 48% |
| v3_turbo_temp0 | 8% | 6% | 14% | 27% | 12% |
| v3_turbo_beam5 | 25% | 33% | 26% | 42% | 46% |
| v3_turbo_nocond | 19% | 31% | 17% | 40% | 29% |