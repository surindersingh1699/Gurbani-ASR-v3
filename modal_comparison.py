"""Modal — Whisper model comparison.

Runs 7 configs (large-v2, large-v3, large-v3-turbo × 5 variants) on 4 kirtan tracks.
Results pushed to HF Hub as named splits. Analysis saved locally.

Usage:
    modal run modal_comparison.py
    modal run modal_comparison.py --model-key large-v2       # single model
    modal run modal_comparison.py --analyze-only             # analyze existing HF splits
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

app = modal.App("surt-model-comparison", image=image)

# ─── Secrets ─────────────────────────────────────────────────────────────────

hf_secret = modal.Secret.from_dotenv(
    path="/Users/surindersingh/Developer/Gurbani_ASR_v3/.env",
)

# ─── Constants ───────────────────────────────────────────────────────────────

HF_REPO = "surindersinghssj/gurbani-asr-model-comparison"
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"
TARGET_SR = 16000

COMMON_PARAMS = dict(
    language="pa",
    word_timestamps=True,
    vad_filter=False,
    initial_prompt=MOOL_MANTAR,
)

CONFIG_OVERRIDES = {
    "large_v2":        {"beam_size": 5, "condition_on_previous_text": True},
    "large_v3":        {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_base":   {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_rep":    {"beam_size": 5, "condition_on_previous_text": True, "repetition_penalty": 1.1},
    "v3_turbo_temp0":  {"beam_size": 5, "condition_on_previous_text": True, "temperature": 0},
    "v3_turbo_beam5":  {"beam_size": 5, "condition_on_previous_text": True},
    "v3_turbo_nocond": {"beam_size": 5, "condition_on_previous_text": False},
}

MODELS = [
    {"key": "large-v2",       "repo": "Systran/faster-whisper-large-v2",
     "configs": ["large_v2"]},
    {"key": "large-v3",       "repo": "Systran/faster-whisper-large-v3",
     "configs": ["large_v3"]},
    {"key": "large-v3-turbo", "repo": "deepdml/faster-whisper-large-v3-turbo-ct2",
     "configs": ["v3_turbo_base", "v3_turbo_rep", "v3_turbo_temp0",
                 "v3_turbo_beam5", "v3_turbo_nocond"]},
]

ALL_CONFIGS = ["large_v2", "large_v3", "v3_turbo_base", "v3_turbo_rep",
               "v3_turbo_temp0", "v3_turbo_beam5", "v3_turbo_nocond"]

# ─── Modal function ───────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=3600,
    secrets=[hf_secret],
    memory=32768,
)
def run_model(model_repo: str, configs: list[str], catalog: list[dict]) -> dict:
    """Run all configs for one model. Returns {config_name: report_dict}."""
    import subprocess, tempfile, time
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

    # Load audio once
    print(f"Loading model {model_repo}...")
    model = WhisperModel(model_repo, device="cuda", compute_type="float16")
    print("Model loaded.")

    audio_cache = {}
    for entry in catalog:
        rec_id = entry["recording_id"]
        try:
            audio_cache[rec_id] = load_audio(entry["url"], rec_id)
            dur = len(audio_cache[rec_id]) / TARGET_SR
            print(f"  Audio {rec_id[:8]}: {dur/60:.1f}m")
        except Exception as e:
            print(f"  ERROR loading {rec_id[:8]}: {e}")

    results = {}
    for config_name in configs:
        print(f"\n--- {config_name} ---")
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
                })

        # Aggregate
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

        results[config_name] = {
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
        agg = results[config_name]["aggregate"]
        print(f"  segs={agg['total_segments']} match={agg['mean_match_score']:.3f} "
              f"conf={agg['mean_confidence']:.3f} cov={agg['tuk_coverage']*100:.1f}%")

    return results


# ─── HF push ─────────────────────────────────────────────────────────────────

def push_to_hf(all_results: dict):
    from datasets import Dataset, Features, Value

    hf_token = os.environ.get("HF_TOKEN")
    for config_name, report in all_results.items():
        raw = report.get("segments", [])
        if not raw:
            print(f"  No segments for {config_name}, skipping HF push")
            continue

        ds_dict = {k: [s[k] for s in raw] for k in raw[0]}
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
        })
        ds = Dataset.from_dict(ds_dict, features=features)
        print(f"  Pushing {len(raw)} rows → split={config_name}")
        ds.push_to_hub(HF_REPO, split=config_name, token=hf_token)
        print(f"  ✓ {config_name}")


# ─── Analysis ─────────────────────────────────────────────────────────────────

def generate_markdown(all_results: dict) -> str:
    from datetime import datetime, timezone
    lines = [
        "# Whisper Model Comparison — Kirtan Analysis",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()}",
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
    for metric, label in [("mean_match_score","Match"), ("mean_confidence","Confidence"), ("tuk_coverage","Coverage")]:
        ranked = sorted(all_results, key=lambda k: all_results[k]["aggregate"].get(metric, 0), reverse=True)
        lines.append(f"- **{label}**: {' > '.join(ranked)}")

    return "\n".join(lines)


# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model_key: str = "", analyze_only: bool = False):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="/Users/surindersingh/Developer/Gurbani_ASR_v3/.env")

    RESULTS_DIR = Path("data/analysis")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "raw_results.json"

    if analyze_only:
        with open(results_path) as f:
            all_results = json.load(f)
    else:
        with open("data/manifests/test_catalog.json") as f:
            catalog = json.load(f)
        print(f"Catalog: {len(catalog)} tracks")

        models_to_run = [m for m in MODELS if not model_key or m["key"] == model_key]

        # Run all models in parallel via Modal .map()
        inputs = [(m["repo"], m["configs"], catalog) for m in models_to_run]
        all_results = {}

        for (repo, configs, _), result in zip(inputs, run_model.starmap(inputs)):
            all_results.update(result)
            for cfg, rep in result.items():
                agg = rep["aggregate"]
                print(f"✓ {cfg}: segs={agg['total_segments']} match={agg['mean_match_score']:.3f} "
                      f"conf={agg['mean_confidence']:.3f} cov={agg['tuk_coverage']*100:.1f}%")

        # Save raw results (without segments to keep file small)
        slim = {k: {kk: vv for kk, vv in v.items() if kk != "segments"}
                for k, v in all_results.items()}
        with open(results_path, "w") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)

        # Push to HF Hub
        print("\nPushing to HF Hub...")
        push_to_hf(all_results)

    # Analysis markdown
    md = generate_markdown(all_results)
    md_path = RESULTS_DIR / "COMPARISON_ANALYSIS.md"
    with open(md_path, "w") as f:
        f.write(md)

    print(f"\n{'='*50}")
    print(f"{'Config':<22} {'Segs':>5} {'Match':>6} {'Conf':>6} {'Cov':>6}")
    print("-" * 50)
    for name in ALL_CONFIGS:
        if name not in all_results:
            continue
        agg = all_results[name]["aggregate"]
        print(f"{name:<22} {agg['total_segments']:>5} {agg['mean_match_score']:>6.3f} "
              f"{agg['mean_confidence']:>6.3f} {agg['tuk_coverage']*100:>5.1f}%")
    print(f"\nAnalysis saved: {md_path}")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO}")
