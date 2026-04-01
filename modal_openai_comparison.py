"""Modal — OpenAI Whisper (PyTorch) model comparison.

Runs original openai-whisper large-v2 and large-v3 on 4 kirtan tracks.
Results pushed as new splits to the same HF repo as the faster-whisper run,
so all configs can be compared side-by-side.

New splits added: openai_large_v2, openai_large_v3

Usage:
    modal run modal_openai_comparison.py
    modal run modal_openai_comparison.py --model-key large-v2   # single model
"""

import json
import os
from pathlib import Path

import modal

# ─── Image ───────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "openai-whisper",
        "torch",
        "soundfile",
        "scipy",
        "numpy",
        "requests",
        "datasets",
        "huggingface-hub",
    )
)

app = modal.App("surt-openai-whisper-comparison", image=image)

# ─── Secrets ─────────────────────────────────────────────────────────────────

hf_secret = modal.Secret.from_dotenv(
    path="/Users/surindersingh/Developer/Gurbani_ASR_v3/.env",
)

# ─── Constants ───────────────────────────────────────────────────────────────

HF_REPO = "surindersinghssj/gurbani-asr-model-comparison"
MOOL_MANTAR = "ੴ ਸਤਿ ਨਾਮੁ ਕਰਤਾ ਪੁਰਖੁ ਨਿਰਭਉ ਨਿਰਵੈਰੁ ਅਕਾਲ ਮੂਰਤਿ ਅਜੂਨੀ ਸੈਭੰ ਗੁਰ ਪ੍ਰਸਾਦਿ"
TARGET_SR = 16000

# Split names for HF — prefixed with "openai_" to distinguish from faster-whisper runs
MODELS = [
    {"key": "large-v2", "name": "large-v2", "split": "openai_large_v2"},
    {"key": "large-v3", "name": "large-v3", "split": "openai_large_v3"},
]

# ─── Modal function ───────────────────────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=3600,
    secrets=[hf_secret],
    memory=32768,
)
def run_openai_model(model_name: str, split_name: str, catalog: list[dict]) -> dict:
    """Run one openai-whisper model over all catalog tracks. Returns report dict."""
    import subprocess
    from statistics import mean, median

    import numpy as np
    import requests
    import soundfile as sf
    import whisper

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

    print(f"Loading openai-whisper {model_name}...")
    model = whisper.load_model(model_name, device="cuda")
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

    all_segs = []
    for entry in catalog:
        rec_id = entry["recording_id"]
        if rec_id not in audio_cache:
            continue

        result = model.transcribe(
            audio_cache[rec_id],
            language="pa",
            word_timestamps=True,
            initial_prompt=MOOL_MANTAR,
            beam_size=5,
            condition_on_previous_text=True,
        )

        # Flatten word-level results from segments
        words = []
        for seg in result["segments"]:
            for w in seg.get("words", []):
                words.append({
                    "word": w["word"].strip(),
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w["probability"],
                })

        lang = result.get("language", "?")
        print(f"  {rec_id[:8]}: {len(words)} words lang={lang}")

        matched = match_tuks(words, entry["shabad_lines"])
        for i, seg in enumerate(matched):
            dur = seg["end"] - seg["start"]
            if dur < 0.5:
                continue
            all_segs.append({
                "config_name": split_name,
                "model_name": f"openai-whisper-{model_name}",
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

    # Per-track aggregate
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

    aggregate = {
        "total_segments": len(all_segs),
        "mean_match_score": round(mean(scores), 4) if scores else 0,
        "median_match_score": round(median(scores), 4) if scores else 0,
        "mean_confidence": round(mean(confs), 4) if confs else 0,
        "tuk_coverage": round(tm / max(1, tt), 3),
    }
    print(f"  segs={aggregate['total_segments']} match={aggregate['mean_match_score']:.3f} "
          f"conf={aggregate['mean_confidence']:.3f} cov={aggregate['tuk_coverage']*100:.1f}%")

    return {
        "split_name": split_name,
        "model_name": f"openai-whisper-{model_name}",
        "segments": all_segs,
        "tracks": tracks,
        "aggregate": aggregate,
    }


# ─── HF push ─────────────────────────────────────────────────────────────────

def push_to_hf(report: dict):
    from datasets import Dataset, Features, Value

    hf_token = os.environ.get("HF_TOKEN")
    split_name = report["split_name"]
    raw = report.get("segments", [])
    if not raw:
        print(f"  No segments for {split_name}, skipping HF push")
        return

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
    print(f"  Pushing {len(raw)} rows → split={split_name}")
    ds.push_to_hub(HF_REPO, split=split_name, token=hf_token)
    print(f"  ✓ {split_name}")


# ─── Local entrypoint ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main(model_key: str = ""):
    from dotenv import load_dotenv
    load_dotenv(dotenv_path="/Users/surindersingh/Developer/Gurbani_ASR_v3/.env")

    RESULTS_DIR = Path("data/analysis")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open("data/manifests/test_catalog.json") as f:
        catalog = json.load(f)
    print(f"Catalog: {len(catalog)} tracks")

    models_to_run = [m for m in MODELS if not model_key or m["key"] == model_key]
    print(f"Running: {[m['split'] for m in models_to_run]} (2 GPUs in parallel)")

    # Run both models in parallel via starmap
    inputs = [(m["name"], m["split"], catalog) for m in models_to_run]
    all_results = {}

    for (name, split, _), report in zip(inputs, run_openai_model.starmap(inputs)):
        all_results[split] = report
        agg = report["aggregate"]
        print(f"✓ {split}: segs={agg['total_segments']} match={agg['mean_match_score']:.3f} "
              f"conf={agg['mean_confidence']:.3f} cov={agg['tuk_coverage']*100:.1f}%")

    # Save slim results locally
    slim = {k: {kk: vv for kk, vv in v.items() if kk != "segments"}
            for k, v in all_results.items()}
    results_path = RESULTS_DIR / "openai_whisper_results.json"
    with open(results_path, "w") as f:
        json.dump(slim, f, indent=2, ensure_ascii=False)

    # Push to HF
    print("\nPushing to HF Hub...")
    for report in all_results.values():
        push_to_hf(report)

    # Summary table
    print(f"\n{'='*50}")
    print(f"{'Config':<22} {'Segs':>5} {'Match':>6} {'Conf':>6} {'Cov':>6}")
    print("-" * 50)
    for split, report in all_results.items():
        agg = report["aggregate"]
        print(f"{split:<22} {agg['total_segments']:>5} {agg['mean_match_score']:>6.3f} "
              f"{agg['mean_confidence']:>6.3f} {agg['tuk_coverage']*100:>5.1f}%")

    print(f"\nResults saved: {results_path}")
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO}")
