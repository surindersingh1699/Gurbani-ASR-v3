# Whisper Model Comparison — Kirtan Analysis

Generated: 2026-03-28T01:08:59.902967+00:00

| Config | Model | Segments | Match | Confidence | Tuk Coverage |
|--------|-------|----------|-------|------------|--------------|
| large_v2 | faster-whisper-large-v2 | 60 | 0.329 | 0.766 | 19.4% |
| large_v3 | faster-whisper-large-v3 | 114 | 0.296 | 0.915 | 19.4% |
| v3_turbo_base | faster-whisper-large-v3-turbo-ct2 | 95 | 0.284 | 0.798 | 33.3% |
| v3_turbo_rep | faster-whisper-large-v3-turbo-ct2 | 98 | 0.299 | 0.810 | 18.1% |
| v3_turbo_temp0 | faster-whisper-large-v3-turbo-ct2 | 42 | 0.307 | 0.858 | 6.9% |
| v3_turbo_beam5 | faster-whisper-large-v3-turbo-ct2 | 85 | 0.296 | 0.804 | 31.9% |
| v3_turbo_nocond | faster-whisper-large-v3-turbo-ct2 | 80 | 0.259 | 0.800 | 23.6% |

## Rankings
- **Match**: large_v2 > v3_turbo_temp0 > v3_turbo_rep > large_v3 > v3_turbo_beam5 > v3_turbo_base > v3_turbo_nocond
- **Confidence**: large_v3 > v3_turbo_temp0 > v3_turbo_rep > v3_turbo_beam5 > v3_turbo_nocond > v3_turbo_base > large_v2
- **Coverage**: v3_turbo_base > v3_turbo_beam5 > v3_turbo_nocond > large_v2 > large_v3 > v3_turbo_rep > v3_turbo_temp0