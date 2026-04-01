"""Modal — Whisper model comparison v2.

Runs 7 configs (large-v2, large-v3, large-v3-turbo × 5 variants) on 25 style-diverse
kirtan tracks. Each config runs as its own Modal function (parallelism = 7).
Results + audio clips pushed to HF Hub as named splits.

Usage:
    modal run modal_comparison_v2.py
    modal run modal_comparison_v2.py --config-key large_v3    # single config
    modal run modal_comparison_v2.py --analyze-only           # analyze existing HF splits
"""

import json
import os
from pathlib import Path

import modal

# ─── Image ───────────────────────────────────────────────────────────────────

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper>=1.1.0",
        "soundfile",
        "scipy",
        "numpy",
        "requests",
        "datasets",
        "huggingface-hub",
    )
)

app = modal.App("surt-model-comparison-v2", image=image)

# ─── Secrets ─────────────────────────────────────────────────────────────────

hf_secret = modal.Secret.from_dotenv(
    path="/Users/surindersingh/Developer/Gurbani_ASR_v3/.env",
)

# ─── Constants ───────────────────────────────────────────────────────────────

HF_REPO = "surindersinghssj/gurbani-asr-comparison-v2"
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"
TARGET_SR = 16000

COMMON_PARAMS = dict(
    language="pa",
    word_timestamps=True,
    vad_filter=False,
    initial_prompt=MOOL_MANTAR,
)

CONFIG_OVERRIDES = {
    "whisper_base":    {"beam_size": 5, "condition_on_previous_text": True},
    "whisper_small":   {"beam_size": 5, "condition_on_previous_text": True},
    "large_v2":        {"beam_size": 5, "condition_on_previous_text": True},
    "large_v3":        {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_base":   {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_rep":    {"beam_size": 5, "condition_on_previous_text": True, "repetition_penalty": 1.1},
    "v3_turbo_temp0":  {"beam_size": 5, "condition_on_previous_text": True, "temperature": 0},
    "v3_turbo_beam5":  {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_nocond": {"beam_size": 5, "condition_on_previous_text": False},
}

# Maps config_name → model_repo
CONFIG_MODEL = {
    "whisper_base":    "Systran/faster-whisper-base",
    "whisper_small":   "Systran/faster-whisper-small",
    "large_v2":        "Systran/faster-whisper-large-v2",
    "large_v3":        "Systran/faster-whisper-large-v3",
    "v3_turbo_base":   "deepdml/faster-whisper-large-v3-turbo-ct2",
    "v3_turbo_rep":    "deepdml/faster-whisper-large-v3-turbo-ct2",
    "v3_turbo_temp0":  "deepdml/faster-whisper-large-v3-turbo-ct2",
    "v3_turbo_beam5":  "deepdml/faster-whisper-large-v3-turbo-ct2",
    "v3_turbo_nocond": "deepdml/faster-whisper-large-v3-turbo-ct2",
}

ALL_CONFIGS = ["whisper_base", "whisper_small", "large_v2", "large_v3",
               "v3_turbo_base", "v3_turbo_rep", "v3_turbo_temp0",
               "v3_turbo_beam5", "v3_turbo_nocond"]

# ─── Modal function — one config per call ─────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=7200,        # 2 hours per config; 25 tracks × ~15 min avg / ~4x realtime ≈ 90 min
    secrets=[hf_secret],
    memory=32768,
)
def run_config(model_repo: str, config_name: str, catalog: list[dict]) -> dict:
    """Run one config on all catalog tracks. Returns {config_name: report_dict}."""
    import io
    import subprocess
    from statistics import mean, median

    import numpy as np
    import requests
    import soundfile as sf
    from faster_whisper import WhisperModel

    MATRAS = set("ਾਿੀੁੂੇੈੋੌੰੱ਼ਁ")

    def normalize(text):
        return "".join(c for c in text if c not in MATRAS).lower().strip()

    def char_f1(a, b):
        wa, wb = set(a.split()), set(b.split())
        if not wa or not wb:
            return 0.0
        overlap = wa & wb
        if not overlap:
            return 0.0
        p, r = len(overlap) / len(wa), len(overlap) / len(wb)
        return 2 * p * r / (p + r)

    def load_audio(url, rec_id):
        path = f"/tmp/{rec_id}.mp3"
        if not os.path.exists(path):
            resp = requests.get(url, allow_redirects=True, timeout=300,
                                headers={"User-Agent": "GurbaniASR/1.0"})
            resp.raise_for_status()
            with open(path, "wb") as f:
                f.write(resp.content)
        wav = f"/tmp/{rec_id}.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(TARGET_SR), "-ac", "1", wav],
            capture_output=True, check=True,
        )
        audio, _ = sf.read(wav, dtype="float32")
        os.unlink(wav)
        return audio

    def slice_audio_flac(audio_array, start, end):
        s = max(0, int(start * TARGET_SR))
        e = min(len(audio_array), int(end * TARGET_SR))
        buf = io.BytesIO()
        sf.write(buf, audio_array[s:e], TARGET_SR, format="FLAC")
        return buf.getvalue()

    def match_tuks(words, lines, min_score=0.2):
        segs, ptr = [], 0
        for tuk_idx, tuk in enumerate(lines):
            tuk_norm = normalize(tuk)
            n = len(tuk_norm.split())
            if not n:
                continue
            rep, sfrom = 0, max(0, ptr - 3)
            while sfrom < len(words) - max(1, n // 2):
                best, bs, be = 0.0, sfrom, sfrom + 1
                for win in range(max(1, n - 3), min(n * 3 + 3, len(words) - sfrom + 1)):
                    end = sfrom + win
                    if end > len(words):
                        break
                    wt = normalize(" ".join(words[i]["word"] for i in range(sfrom, end)))
                    sc = char_f1(wt, tuk_norm)
                    if sc > best:
                        best, bs, be = sc, sfrom, end
                if best >= min_score:
                    sl = words[bs:be]
                    segs.append({
                        "tuk_index": tuk_idx,
                        "canonical_line": tuk,
                        "whisper_text": " ".join(w["word"] for w in sl),
                        "start": sl[0]["start"], "end": sl[-1]["end"],
                        "match_score": round(best, 4),
                        "avg_confidence": round(mean(w["probability"] for w in sl), 4),
                        "repetition": rep,
                    })
                    sfrom, ptr, rep = be, be, rep + 1
                else:
                    sfrom += max(1, n // 2)
                    if sfrom - ptr > n * 5:
                        break
        return segs

    print(f"Loading model {model_repo} for config={config_name}...")
    model = WhisperModel(model_repo, device="cuda", compute_type="float16")
    print("Model loaded.")

    audio_cache = {}
    for entry in catalog:
        rec_id = entry["recording_id"]
        try:
            audio_cache[rec_id] = load_audio(entry["url"], rec_id)
            dur = len(audio_cache[rec_id]) / TARGET_SR
            print(f"  {rec_id[:8]}: {dur/60:.1f}m")
        except Exception as e:
            print(f"  ERROR {rec_id[:8]}: {e}")

    params = {**COMMON_PARAMS, **CONFIG_OVERRIDES[config_name]}
    all_segs = []

    for entry in catalog:
        rec_id = entry["recording_id"]
        if rec_id not in audio_cache:
            continue
        seg_iter, info = model.transcribe(audio_cache[rec_id], **params)
        words = []
        for seg in seg_iter:
            if seg.words:
                words.extend({"word": w.word.strip(), "start": w.start,
                              "end": w.end, "probability": w.probability}
                             for w in seg.words)
        print(f"  {rec_id[:8]}: {len(words)} words lang={info.language} p={info.language_probability:.2f}")

        matched = match_tuks(words, entry["shabad_lines"])
        for i, seg in enumerate(matched):
            dur = seg["end"] - seg["start"]
            if dur < 0.5:
                continue
            all_segs.append({
                "config_name": config_name,
                "model_name": model_repo.split("/")[-1],
                "recording_id": rec_id,
                "segment_id": f"{rec_id}_{i:04d}",
                "tuk_index": seg["tuk_index"],
                "canonical_line": seg["canonical_line"],
                "whisper_text": seg["whisper_text"],
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "duration": round(dur, 3),
                "match_score": seg["match_score"],
                "avg_confidence": seg["avg_confidence"],
                "repetition": seg["repetition"],
                "ang": entry.get("ang", 0),
                "raag": entry.get("raag", ""),
                "writer": entry.get("writer", ""),
                "style_bucket": entry.get("style_bucket", ""),
                "artist_name": entry.get("artist_name", ""),
                "source_url": entry.get("url", ""),
                "audio_bytes": slice_audio_flac(audio_cache[rec_id], seg["start"], seg["end"]),
            })

    tracks = []
    for entry in catalog:
        rec_id = entry["recording_id"]
        segs = [s for s in all_segs if s["recording_id"] == rec_id]
        matched_tuks = {s["tuk_index"] for s in segs if s["repetition"] == 0}
        n_lines = len(entry["shabad_lines"])
        tracks.append({
            "recording_id": rec_id,
            "style_bucket": entry.get("style_bucket", ""),
            "artist_name": entry.get("artist_name", ""),
            "segments": len(segs),
            "tuks_matched": len(matched_tuks),
            "tuks_total": n_lines,
            "tuk_coverage": round(len(matched_tuks) / max(1, n_lines), 3),
            "mean_match_score": round(mean(s["match_score"] for s in segs), 4) if segs else 0,
            "mean_confidence": round(mean(s["avg_confidence"] for s in segs), 4) if segs else 0,
        })

    scores = [s["match_score"] for s in all_segs]
    confs  = [s["avg_confidence"] for s in all_segs]
    tm = sum(t["tuks_matched"] for t in tracks)
    tt = sum(t["tuks_total"] for t in tracks)

    report = {
        "config_name": config_name,
        "model_name": model_repo.split("/")[-1],
        "segments": all_segs,
        "tracks": tracks,
        "aggregate": {
            "total_segments": len(all_segs),
            "mean_match_score": round(mean(scores), 4) if scores else 0,
            "median_match_score": round(median(scores), 4) if scores else 0,
            "mean_confidence": round(mean(confs), 4) if confs else 0,
            "tuk_coverage": round(tm / max(1, tt), 3),
        },
    }
    agg = report["aggregate"]
    print(f"  [{config_name}] segs={agg['total_segments']} match={agg['mean_match_score']:.3f} "
          f"conf={agg['mean_confidence']:.3f} cov={agg['tuk_coverage']*100:.1f}%")
    return {config_name: report}


# ─── HF push ─────────────────────────────────────────────────────────────────

def push_to_hf(all_results: dict):
    # Audio encoding with datasets Audio() requires torchcodec (torch).
    # Store audio as raw FLAC bytes in a binary column to avoid the dependency.
    # The bytes are playable — just not auto-rendered in the HF dataset viewer.
    from datasets import Dataset, Features, Value

    hf_token = os.environ.get("HF_TOKEN")
    for config_name, report in all_results.items():
        raw = report.get("segments", [])
        if not raw:
            print(f"  No segments for {config_name}, skipping")
            continue

        ds_dict = {k: [s[k] for s in raw] for k in raw[0] if k != "audio_bytes"}
        # Store FLAC audio as raw bytes (Arrow binary). No torch/torchcodec needed.
        ds_dict["audio_flac"] = [s["audio_bytes"] for s in raw]

        features = Features({
            "config_name": Value("string"), "model_name": Value("string"),
            "recording_id": Value("string"), "segment_id": Value("string"),
            "tuk_index": Value("int32"), "canonical_line": Value("string"),
            "whisper_text": Value("string"), "start": Value("float32"),
            "end": Value("float32"), "duration": Value("float32"),
            "match_score": Value("float32"), "avg_confidence": Value("float32"),
            "repetition": Value("int32"), "ang": Value("int32"),
            "raag": Value("string"), "writer": Value("string"),
            "style_bucket": Value("string"), "artist_name": Value("string"),
            "source_url": Value("string"),
            "audio_flac": Value("binary"),
        })
        ds = Dataset.from_dict(ds_dict, features=features)
        print(f"  Pushing {len(raw)} rows → split={config_name}")
        ds.push_to_hub(HF_REPO, split=config_name, token=hf_token)
        print(f"  ✓ {config_name}")


# ─── Analysis ─────────────────────────────────────────────────────────────────

def generate_markdown(all_results: dict) -> str:
    from datetime import datetime, timezone
    lines = [
        "# Whisper Model Comparison v2 — Style-Diverse Kirtan Analysis",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
        "\n25 tracks across 5 style buckets (hazoori, puratan, akj, live, mixed)",
        "\n| Config | Model | Segments | Match | Confidence | Tuk Coverage |",
        "|--------|-------|----------|-------|------------|--------------|",
    ]
    for name in ALL_CONFIGS:
        if name not in all_results:
            continue
        agg = all_results[name]["aggregate"]
        lines.append(f"| {name} | {all_results[name]['model_name']} | "
                     f"{agg['total_segments']} | {agg['mean_match_score']:.3f} | "
                     f"{agg['mean_confidence']:.3f} | {agg['tuk_coverage']*100:.1f}% |")

    lines.append("\n## Rankings")
    for metric, label in [("mean_match_score", "Match"), ("mean_confidence", "Confidence"),
                           ("tuk_coverage", "Coverage")]:
        ranked = sorted(all_results, key=lambda k: all_results[k]["aggregate"].get(metric, 0),
                        reverse=True)
        lines.append(f"- **{label}**: {' > '.join(ranked)}")

    lines.append("\n## Per-Bucket Coverage")
    lines.append("| Config | hazoori | puratan | akj | live | mixed |")
    lines.append("|--------|---------|---------|-----|------|-------|")
    for name in ALL_CONFIGS:
        if name not in all_results:
            continue
        tracks = all_results[name].get("tracks", [])
        row = [name]
        for bucket in ["hazoori", "puratan", "akj", "live", "mixed"]:
            bt = [t for t in tracks if t["style_bucket"] == bucket]
            if bt:
                cov = sum(t["tuks_matched"] for t in bt) / max(1, sum(t["tuks_total"] for t in bt))
                row.append(f"{cov*100:.0f}%")
            else:
                row.append("—")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(config_key: str = "", analyze_only: bool = False):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="/Users/surindersingh/Developer/Gurbani_ASR_v3/.env")

    RESULTS_DIR = Path("data/analysis_v2")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "raw_results.json"

    if analyze_only:
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        with open("data/manifests/test_catalog_v2.json") as f:
            catalog = json.load(f)

        configs_to_run = [c for c in ALL_CONFIGS if not config_key or c == config_key]
        print(f"Catalog v2: {len(catalog)} tracks | Configs: {configs_to_run}")

        # One function call per config — all run in parallel
        inputs = [(CONFIG_MODEL[c], c, catalog) for c in configs_to_run]
        all_results = {}

        for result in run_config.starmap(inputs):
            all_results.update(result)
            for cfg, rep in result.items():
                agg = rep["aggregate"]
                print(f"✓ {cfg}: segs={agg['total_segments']} match={agg['mean_match_score']:.3f} "
                      f"conf={agg['mean_confidence']:.3f} cov={agg['tuk_coverage']*100:.1f}%")

        slim = {k: {kk: vv for kk, vv in v.items() if kk != "segments"}
                for k, v in all_results.items()}
        with open(results_path, "w") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)

        print("\nPushing to HF Hub...")
        push_to_hf(all_results)

    md = generate_markdown(all_results)
    md_path = RESULTS_DIR / "COMPARISON_ANALYSIS.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n{'='*60}")
    print(f"{'Config':<22} {'Segs':>5} {'Match':>6} {'Conf':>6} {'Cov':>6}")
    print("-" * 60)
    for name in ALL_CONFIGS:
        if name not in all_results:
            continue
        agg = all_results[name]["aggregate"]
        print(f"{name:<22} {agg['total_segments']:>5} {agg['mean_match_score']:>6.3f} "
              f"{agg['mean_confidence']:>6.3f} {agg['tuk_coverage']*100:>5.1f}%")
    print(f"\nAnalysis saved: {md_path}")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO}")
